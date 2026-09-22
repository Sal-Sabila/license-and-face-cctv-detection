-- MySQL dump 10.13  Distrib 8.4.11, for Linux (x86_64)
--
-- Host: localhost    Database: real_cctv
-- ------------------------------------------------------
-- Server version	8.4.11-0ubuntu0.26.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `cameras`
--

DROP TABLE IF EXISTS `cameras`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `cameras` (
  `camera_id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `location` varchar(150) DEFAULT NULL,
  `stream_url` text,
  `stream_type` tinyint DEFAULT NULL,
  `status` tinyint DEFAULT NULL,
  `direction` enum('entry','exit','unknown') NOT NULL DEFAULT 'unknown',
  PRIMARY KEY (`camera_id`),
  KEY `idx_cameras_direction` (`direction`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `full_detection`
--

DROP TABLE IF EXISTS `full_detection`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `full_detection` (
  `detection_id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `plate_id` bigint unsigned DEFAULT NULL,
  `camera_id` bigint unsigned DEFAULT NULL,
  `detection_status` tinyint NOT NULL,
  `detection_confidence` decimal(6,5) DEFAULT NULL,
  `face_image_path` varchar(500) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `track_id` bigint DEFAULT NULL,
  `object_type` enum('vehicle','person') NOT NULL DEFAULT 'person',
  `vehicle_type` enum('car','motorcycle','truck','bus','unknown') NOT NULL DEFAULT 'unknown',
  `has_plate` tinyint(1) NOT NULL DEFAULT '0',
  `has_driver` tinyint(1) NOT NULL DEFAULT '0',
  `vehicle_confidence` decimal(6,5) DEFAULT NULL,
  `plate_detection_confidence` decimal(6,5) DEFAULT NULL,
  `ocr_confidence` decimal(6,5) DEFAULT NULL,
  `person_confidence` decimal(6,5) DEFAULT NULL,
  `face_confidence` decimal(6,5) DEFAULT NULL,
  `direction` enum('entry','exit','unknown') NOT NULL DEFAULT 'unknown',
  `driver_track_id` bigint DEFAULT NULL,
  `driver_face_path` varchar(500) DEFAULT NULL,
  `event_key` varchar(160) DEFAULT NULL,
  `vehicle_image_path` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`detection_id`),
  KEY `full_detection_plate_id_foreign` (`plate_id`),
  KEY `idx_detection_event_type` (`object_type`,`created_at`),
  KEY `idx_detection_track` (`camera_id`,`track_id`,`direction`,`created_at`),
  KEY `idx_detection_event_key` (`event_key`),
  CONSTRAINT `full_detection_plate_id_foreign` FOREIGN KEY (`plate_id`) REFERENCES `plate` (`plate_id`)
) ENGINE=InnoDB AUTO_INCREMENT=9516 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `plate`
--

DROP TABLE IF EXISTS `plate`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `plate` (
  `plate_id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `plate_number` varchar(10) DEFAULT NULL,
  `detection_status` tinyint NOT NULL,
  `detection_confidence` decimal(6,5) DEFAULT NULL,
  `ocr_confidence` decimal(6,5) DEFAULT NULL,
  `plate_image_path` varchar(500) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `raw_ocr_text` varchar(100) DEFAULT NULL,
  `normalized_plate_number` varchar(30) DEFAULT NULL,
  PRIMARY KEY (`plate_id`),
  KEY `plate_plate_number_index` (`plate_number`)
) ENGINE=InnoDB AUTO_INCREMENT=1487 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `plate_logs`
--

DROP TABLE IF EXISTS `plate_logs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `plate_logs` (
  `plate_id` bigint unsigned DEFAULT NULL,
  `camera_id` bigint unsigned DEFAULT NULL,
  `status` tinyint NOT NULL,
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  KEY `plate_logs_camera_id_created_at_index` (`camera_id`,`created_at`),
  KEY `plate_logs_plate_id_foreign` (`plate_id`),
  CONSTRAINT `plate_logs_camera_id_foreign` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`camera_id`),
  CONSTRAINT `plate_logs_plate_id_foreign` FOREIGN KEY (`plate_id`) REFERENCES `plate` (`plate_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `suspicious_plates`
--

DROP TABLE IF EXISTS `suspicious_plates`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `suspicious_plates` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `plate_id` bigint unsigned NOT NULL,
  `camera_id` bigint unsigned DEFAULT NULL,
  `first_detected_at` datetime(3) NOT NULL,
  `last_detected_at` datetime(3) NOT NULL,
  `entry_count` int NOT NULL,
  `exit_count` int NOT NULL,
  `total_activity` int NOT NULL,
  `status` enum('active','resolved','ignored') DEFAULT 'active',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `suspicious_plates_plate_id_index` (`plate_id`),
  KEY `suspicious_plates_camera_id_foreign` (`camera_id`),
  CONSTRAINT `suspicious_plates_camera_id_foreign` FOREIGN KEY (`camera_id`) REFERENCES `cameras` (`camera_id`),
  CONSTRAINT `suspicious_plates_plate_id_foreign` FOREIGN KEY (`plate_id`) REFERENCES `plate` (`plate_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `system_settings`
--

DROP TABLE IF EXISTS `system_settings`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `system_settings` (
  `setting_key` varchar(100) NOT NULL,
  `setting_value` text,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`setting_key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping routines for database 'real_cctv'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-09-22 11:09:47
