import collections
import concurrent.futures
import os
import threading
import time
from dataclasses import dataclass

import psutil

from ai.plate.ocr import PlateOCR


STREAM_QUEUE_SIZE = max(1, int(os.environ.get("PLATE_OCR_STREAM_QUEUE_SIZE", "8")))
SEGMENT_QUEUE_SIZE = max(1, int(os.environ.get("PLATE_OCR_SEGMENT_QUEUE_SIZE", "16")))
STREAM_MAX_AGE_SECONDS = max(
    0.1, float(os.environ.get("PLATE_OCR_STREAM_MAX_AGE_SECONDS", "5.0"))
)


@dataclass
class _OCRTask:
    image: object
    future: concurrent.futures.Future
    source: str
    submitted_at: float
    coalesce_key: object = None
    retry: bool = False


class PlateOCRLane:
    def __init__(self, manager, source):
        self.manager = manager
        self.source = source

    def warmup(self):
        return self.manager.warmup()

    def submit(self, image, coalesce_key=None, retry=False):
        return self.manager.submit(
            self.source,
            image,
            coalesce_key=coalesce_key,
            retry=retry,
        )

    def stats(self):
        return self.manager.stats(self.source)


class SharedPlateOCRWorkers:
    """Two bounded OCR lanes sharing one PaddleOCR model instance."""

    def __init__(
        self,
        ocr_factory=None,
        stream_queue_size=STREAM_QUEUE_SIZE,
        segment_queue_size=SEGMENT_QUEUE_SIZE,
        stream_max_age=STREAM_MAX_AGE_SECONDS,
    ):
        self._ocr_factory = ocr_factory or (
            lambda: PlateOCR(
                scale=2.5,
                min_confidence=0.15,
                verbose=False,
                fast_mode=True,
            )
        )
        self._queue_limits = {
            "stream": max(1, int(stream_queue_size)),
            "segment": max(1, int(segment_queue_size)),
        }
        self._stream_max_age = max(0.1, float(stream_max_age))
        self._queues = {
            "stream": collections.deque(),
            "segment": collections.deque(),
        }
        self._condition = threading.Condition()
        self._ocr = None
        self._ocr_load_lock = threading.Lock()
        self._workers_started = False
        self._worker_threads = []
        self._stopping = False
        self._active_source = None
        self._segment_variant_cursor = 0
        self._stream_variant_cursor = 0
        self._queue_wait_samples = {
            lane: collections.deque(maxlen=256) for lane in self._queues
        }
        self._total_samples = {
            lane: collections.deque(maxlen=256) for lane in self._queues
        }
        self._stats = {
            lane: {
                "completed": 0,
                "submitted": 0,
                "dropped": 0,
                "stale": 0,
                "errors": 0,
                "retries": 0,
                "recognized_reads": 0,
                "backlog_peak": 0,
                "ocr_seconds_total": 0.0,
                "ocr_seconds_last": 0.0,
                "task_seconds_total": 0.0,
                "task_seconds_last": 0.0,
                "queue_wait_ms_last": 0.0,
                "queue_wait_ms_max": 0.0,
                "total_seconds_last": 0.0,
                "total_seconds_max": 0.0,
            }
            for lane in self._queues
        }
        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(interval=None)

    def lane(self, source):
        if source not in self._queues:
            raise ValueError(f"Unsupported OCR lane: {source}")
        return PlateOCRLane(self, source)

    def warmup(self):
        with self._ocr_load_lock:
            if self._ocr is None:
                self._ocr = self._ocr_factory()
                print("[OCR WORKERS] Satu instance PaddleOCR berhasil dibuat")
                warmup = getattr(self._ocr, "warmup", None)
                if warmup:
                    warmup()
        return self._ocr

    def _start_workers(self):
        with self._condition:
            if self._workers_started:
                return
            self._workers_started = True
            for source in ("stream", "segment"):
                worker = threading.Thread(
                    target=self._worker_loop,
                    args=(source,),
                    name=f"PlateOCR-{source.title()}",
                    daemon=True,
                )
                self._worker_threads.append(worker)
                worker.start()
            print(
                "[OCR WORKERS] Workers stream+segment aktif; "
                f"queue_limits={self._queue_limits}"
            )

    def submit(self, source, image, coalesce_key=None, retry=False):
        if source not in self._queues:
            raise ValueError(f"Unsupported OCR lane: {source}")
        if image is None or getattr(image, "size", 0) == 0:
            return None

        self._start_workers()
        future = concurrent.futures.Future()
        task = _OCRTask(
            image=image.copy(),
            future=future,
            source=source,
            submitted_at=time.perf_counter(),
            coalesce_key=coalesce_key,
            retry=bool(retry),
        )
        with self._condition:
            lane_queue = self._queues[source]
            if source == "stream" and coalesce_key is not None:
                retained = collections.deque()
                for queued in lane_queue:
                    if queued.coalesce_key == coalesce_key:
                        if queued.future.cancel():
                            self._stats[source]["dropped"] += 1
                            print(
                                "[OCR COALESCE] source=stream "
                                "superseded stale sample for same track"
                            )
                    else:
                        retained.append(queued)
                self._queues[source] = lane_queue = retained

            while (
                source == "segment"
                and len(lane_queue) >= self._queue_limits[source]
                and not self._stopping
            ):
                self._condition.wait()
                lane_queue = self._queues[source]

            if self._stopping:
                raise RuntimeError("OCR worker scheduler is shutting down")

            if len(lane_queue) >= self._queue_limits[source]:
                if source == "stream":
                    dropped = lane_queue.popleft()
                    dropped.future.cancel()
                    self._stats[source]["dropped"] += 1
                    print(
                        "[OCR DROP] source=stream reason=queue_full "
                        f"limit={self._queue_limits[source]}"
                    )
                else:
                    raise RuntimeError(
                        "Segment OCR queue capacity invariant was violated"
                    )

            lane_queue.append(task)
            self._condition.notify_all()
            backlog = len(lane_queue)
            self._stats[source]["backlog_peak"] = max(
                self._stats[source]["backlog_peak"],
                backlog,
            )
            self._stats[source]["submitted"] += 1
        print(
            f"[OCR QUEUE] source={source} backlog={backlog}/"
            f"{self._queue_limits[source]}"
        )
        return future

    def _next_task(self, source):
        with self._condition:
            while True:
                if self._stopping:
                    return None
                stream_waiting = bool(self._queues["stream"])
                segment_waiting = bool(self._queues["segment"])
                can_run = self._active_source is None
                if source == "stream":
                    can_run = can_run and stream_waiting
                else:
                    can_run = can_run and segment_waiting and not stream_waiting

                if can_run:
                    task = self._queues[source].popleft()
                    if not task.future.set_running_or_notify_cancel():
                        continue
                    self._active_source = source
                    self._condition.notify_all()
                    return task
                self._condition.wait()

    def _worker_loop(self, source):
        while True:
            task = self._next_task(source)
            if task is None:
                return

            if (
                source == "stream"
                and time.perf_counter() - task.submitted_at > self._stream_max_age
            ):
                with self._condition:
                    self._stats[source]["stale"] += 1
                task.future.set_result(None)
                print(
                    f"[OCR STALE] source=stream age="
                    f"{time.perf_counter() - task.submitted_at:.3f}s"
                )
                with self._condition:
                    self._active_source = None
                    self._condition.notify_all()
                continue

            queue_wait_ms = (time.perf_counter() - task.submitted_at) * 1000.0
            started = time.perf_counter()
            ocr_seconds = 0.0
            try:
                self.warmup()
                read_fast = getattr(self._ocr, "read_fast", None)
                if source == "segment" and callable(read_fast):
                    with self._condition:
                        if source == "stream":
                            variant_start = self._stream_variant_cursor
                            self._stream_variant_cursor += 1
                        else:
                            variant_start = self._segment_variant_cursor
                            self._segment_variant_cursor += 1
                    read_started = time.perf_counter()
                    try:
                        result = read_fast(
                            task.image,
                            variant_index=variant_start,
                        )
                    except Exception:
                        ocr_seconds += time.perf_counter() - read_started
                        raise
                    ocr_seconds += time.perf_counter() - read_started
                else:
                    read_started = time.perf_counter()
                    result = self._ocr.read(task.image)
                    ocr_seconds += time.perf_counter() - read_started
                task_seconds = time.perf_counter() - started
                total_seconds = time.perf_counter() - task.submitted_at
                with self._condition:
                    lane_stats = self._stats[source]
                    lane_stats["completed"] += 1
                    lane_stats["ocr_seconds_last"] = ocr_seconds
                    lane_stats["ocr_seconds_total"] += ocr_seconds
                    lane_stats["task_seconds_last"] = task_seconds
                    lane_stats["task_seconds_total"] += task_seconds
                    lane_stats["queue_wait_ms_last"] = queue_wait_ms
                    lane_stats["queue_wait_ms_max"] = max(
                        lane_stats["queue_wait_ms_max"],
                        queue_wait_ms,
                    )
                    lane_stats["total_seconds_last"] = total_seconds
                    lane_stats["total_seconds_max"] = max(
                        lane_stats["total_seconds_max"],
                        total_seconds,
                    )
                    self._queue_wait_samples[source].append(queue_wait_ms)
                    self._total_samples[source].append(total_seconds)
                    if task.retry:
                        lane_stats["retries"] += 1
                    if result and result.get("text"):
                        lane_stats["recognized_reads"] += 1
                if result is not None:
                    result = dict(result)
                    result["elapsed"] = ocr_seconds
                task.future.set_result(result)
                print(
                    f"[OCR DONE] source={source} queue_wait={queue_wait_ms:.1f}ms "
                    f"ocr={ocr_seconds:.3f}s task={task_seconds:.3f}s "
                    f"total={total_seconds:.3f}s retry={int(task.retry)} "
                    f"text="
                    f"'{(result or {}).get('text', '') or '-'}'"
                )
            except Exception as exc:
                with self._condition:
                    self._stats[source]["errors"] += 1
                    if task.retry:
                        self._stats[source]["retries"] += 1
                task.future.set_exception(exc)
                print(
                    f"[OCR ERROR] source={source} retry={int(task.retry)} "
                    f"queue_wait={queue_wait_ms:.1f}ms "
                    f"elapsed={time.perf_counter() - started:.3f}s error={exc}"
                )
            finally:
                with self._condition:
                    self._active_source = None
                    self._condition.notify_all()

    def shutdown(self, timeout=2.0):
        with self._condition:
            self._stopping = True
            for lane_queue in self._queues.values():
                while lane_queue:
                    lane_queue.popleft().future.cancel()
            self._condition.notify_all()
        for worker in getattr(self, "_worker_threads", ()):
            worker.join(timeout=timeout)

    def stats(self, source=None):
        sources = (source,) if source else tuple(self._queues)
        try:
            cpu_percent = self._process.cpu_percent(interval=None)
            memory_mb = self._process.memory_info().rss / (1024 * 1024)
        except psutil.Error as exc:
            cpu_percent = None
            memory_mb = None
            print(f"[OCR METRICS ERROR] {exc}")

        with self._condition:
            result = {}
            for lane in sources:
                item = dict(self._stats[lane])
                item.update({
                    "backlog": len(self._queues[lane]),
                    "pending": len(self._queues[lane]),
                    "processing": int(self._active_source == lane),
                    "failed": item["errors"],
                    "queue_limit": self._queue_limits[lane],
                    "active": self._active_source == lane,
                    "cpu_percent": cpu_percent,
                    "process_rss_mb": round(memory_mb, 1) if memory_mb is not None else None,
                })
                completed = item["completed"]
                item["ocr_seconds_avg"] = (
                    item["ocr_seconds_total"] / completed if completed else 0.0
                )
                item["task_seconds_avg"] = (
                    item["task_seconds_total"] / completed if completed else 0.0
                )
                item["queue_wait_ms_avg_recent"] = (
                    sum(self._queue_wait_samples[lane])
                    / len(self._queue_wait_samples[lane])
                    if self._queue_wait_samples[lane]
                    else 0.0
                )
                item["total_seconds_avg_recent"] = (
                    sum(self._total_samples[lane]) / len(self._total_samples[lane])
                    if self._total_samples[lane]
                    else 0.0
                )
                result[lane] = item
            return result[source] if source else result


_shared_manager = None
_shared_manager_lock = threading.Lock()


def get_shared_ocr_lane(source):
    global _shared_manager
    with _shared_manager_lock:
        if _shared_manager is None:
            _shared_manager = SharedPlateOCRWorkers()
        return _shared_manager.lane(source)
