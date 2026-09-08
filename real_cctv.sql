-- phpMyAdmin SQL Dump
-- version 5.2.3
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Sep 08, 2026 at 08:48 AM
-- Server version: 8.4.3
-- PHP Version: 8.3.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `real_cctv`
--

-- --------------------------------------------------------

--
-- Table structure for table `cameras`
--

CREATE TABLE `cameras` (
  `camera_id` bigint UNSIGNED NOT NULL,
  `location` varchar(150) DEFAULT NULL,
  `stream_url` text,
  `stream_type` tinyint DEFAULT NULL,
  `status` tinyint DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `full_detection`
--

CREATE TABLE `full_detection` (
  `detection_id` bigint UNSIGNED NOT NULL,
  `plate_id` bigint UNSIGNED NOT NULL,
  `detection_status` tinyint NOT NULL,
  `detection_confidence` decimal(6,5) DEFAULT NULL,
  `face_image_path` varchar(500) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `plate`
--

CREATE TABLE `plate` (
  `plate_id` bigint UNSIGNED NOT NULL,
  `plate_number` varchar(10) DEFAULT NULL,
  `detection_status` tinyint NOT NULL,
  `detection_confidence` decimal(6,5) DEFAULT NULL,
  `ocr_confidence` decimal(6,5) DEFAULT NULL,
  `plate_image_path` varchar(500) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `plate_logs`
--

CREATE TABLE `plate_logs` (
  `plate_id` bigint UNSIGNED DEFAULT NULL,
  `camera_id` bigint UNSIGNED DEFAULT NULL,
  `status` enum('success','failed','warning','info') NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `suspicious_plates`
--

CREATE TABLE `suspicious_plates` (
  `id` bigint UNSIGNED NOT NULL,
  `plate_id` bigint UNSIGNED NOT NULL,
  `camera_id` bigint UNSIGNED DEFAULT NULL,
  `first_detected_at` datetime(3) NOT NULL,
  `last_detected_at` datetime(3) NOT NULL,
  `entry_count` int NOT NULL,
  `exit_count` int NOT NULL,
  `total_activity` int NOT NULL,
  `status` enum('active','resolved','ignored') DEFAULT 'active',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `cameras`
--
ALTER TABLE `cameras`
  ADD PRIMARY KEY (`camera_id`);

--
-- Indexes for table `full_detection`
--
ALTER TABLE `full_detection`
  ADD PRIMARY KEY (`detection_id`),
  ADD KEY `full_detection_plate_id_foreign` (`plate_id`);

--
-- Indexes for table `plate`
--
ALTER TABLE `plate`
  ADD PRIMARY KEY (`plate_id`),
  ADD KEY `plate_plate_number_index` (`plate_number`);

--
-- Indexes for table `plate_logs`
--
ALTER TABLE `plate_logs`
  ADD KEY `plate_logs_camera_id_created_at_index` (`camera_id`,`created_at`),
  ADD KEY `plate_logs_plate_id_foreign` (`plate_id`);

--
-- Indexes for table `suspicious_plates`
--
ALTER TABLE `suspicious_plates`
  ADD PRIMARY KEY (`id`),
  ADD KEY `suspicious_plates_plate_id_index` (`plate_id`),
  ADD KEY `suspicious_plates_camera_id_foreign` (`camera_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `cameras`
--
ALTER TABLE `cameras`
  MODIFY `camera_id` bigint UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `full_detection`
--
ALTER TABLE `full_detection`
  MODIFY `detection_id` bigint UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `plate`
--
ALTER TABLE `plate`
  MODIFY `plate_id` bigint UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `suspicious_plates`
--
ALTER TABLE `suspicious_plates`
  MODIFY `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `full_detection`
--
ALTER TABLE `full_detection`
  ADD CONSTRAINT `full_detection_plate_id_foreign` FOREIGN KEY (`plate_id`) REFERENCES `plate` (`plate_id`);

--
-- Constraints for table `plate_logs`
--
ALTER TABLE `plate_logs`
  ADD CONSTRAINT `plate_logs_camera_id_foreign` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`camera_id`),
  ADD CONSTRAINT `plate_logs_plate_id_foreign` FOREIGN KEY (`plate_id`) REFERENCES `plate` (`plate_id`);

--
-- Constraints for table `suspicious_plates`
--
ALTER TABLE `suspicious_plates`
  ADD CONSTRAINT `suspicious_plates_camera_id_foreign` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`camera_id`),
  ADD CONSTRAINT `suspicious_plates_plate_id_foreign` FOREIGN KEY (`plate_id`) REFERENCES `plate` (`plate_id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
