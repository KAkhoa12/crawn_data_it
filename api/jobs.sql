-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Máy chủ: 127.0.0.1
-- Thời gian đã tạo: Th6 19, 2025 lúc 03:57 PM
-- Phiên bản máy phục vụ: 10.4.32-MariaDB-log
-- Phiên bản PHP: 8.1.25

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Cơ sở dữ liệu: `jobs`
--

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `answer`
--

CREATE TABLE `answer` (
  `id` int(11) NOT NULL,
  `questionID` int(11) NOT NULL,
  `content` text NOT NULL,
  `is_correct` tinyint(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `answer`
--

INSERT INTO `answer` (`id`, `questionID`, `content`, `is_correct`) VALUES
(1, 1, '2', 1),
(2, 1, '3', 0),
(5, 1, '4', 0),
(6, 1, '5', 0),
(8, 4, 'Tố Hữu', 1),
(9, 4, 'Xuân Diệu', 0),
(10, 4, 'Quang Trung', 0),
(11, 5, 'Đúng', 1),
(12, 5, 'Sai', 0),
(13, 7, 'Tú', 0),
(14, 7, 'World', 1),
(15, 7, 'Các bạn', 0),
(16, 7, 'Gì', 0);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `application`
--

CREATE TABLE `application` (
  `id` int(11) NOT NULL,
  `job_id` int(11) NOT NULL,
  `seeker_id` int(11) NOT NULL,
  `applied_at` datetime NOT NULL,
  `status` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `application`
--

INSERT INTO `application` (`id`, `job_id`, `seeker_id`, `applied_at`, `status`) VALUES
(88, 101, 65, '2025-04-13 04:10:48', 2),
(89, 101, 65, '2025-04-13 04:15:14', 2),
(90, 101, 65, '2025-04-13 04:20:45', 2),
(91, 89, 65, '2025-04-13 04:27:41', 0),
(92, 89, 65, '2025-04-13 04:28:12', 0),
(94, 31, 65, '2025-01-27 11:36:30', 1),
(95, 31, 65, '2025-01-27 11:36:30', 1),
(97, 31, 65, '2025-01-27 18:36:30', 0),
(98, 142, 65, '2025-05-06 04:57:51', 0),
(99, 140, 95, '2025-06-01 07:59:14', 0),
(100, 195, 95, '2025-06-08 02:49:01', 3);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `category`
--

CREATE TABLE `category` (
  `id` int(11) NOT NULL,
  `category_name` varchar(500) NOT NULL,
  `subcategory_name` varchar(500) NOT NULL,
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `category`
--

INSERT INTO `category` (`id`, `category_name`, `subcategory_name`, `status`) VALUES
(1, 'Software Engineering', 'Backend Developer', 1),
(2, 'Software Engineering', 'Frontend Developer', 1),
(3, 'Software Engineering', 'Fullstack Developer', 1),
(4, 'Software Engineering', 'Mobile Developer', 1),
(5, 'Software Engineering', 'Web Developer', 1),
(6, 'Software Engineering', 'DevOps Engineer', 1),
(7, 'Software Engineering', 'Game Developer', 1),
(8, 'Software Engineering', 'Embedded Systems Engineer', 1),
(9, 'Software Engineering', 'Cloud Engineer', 1),
(10, 'Software Engineering', 'Software Architect', 1),
(11, 'Software Engineering', 'Software Development Engineer in Test (SDET)', 1),
(12, 'Software Engineering', 'UI/UX Developer', 1),
(13, 'Software Engineering', 'Blockchain Developer', 1),
(14, 'Software Engineering', 'Machine Learning Engineer', 1),
(15, 'Software Testing', 'Software Tester', 1),
(16, 'Software Testing', 'Automation Tester', 1),
(17, 'Software Testing', 'Manual Tester', 1),
(18, 'Software Testing', 'Performance Tester', 1),
(19, 'Software Testing', 'QA Engineer', 1),
(20, 'Software Testing', 'Game Tester', 1),
(21, 'Software Testing', 'Security Tester', 1),
(22, 'Artificial Intelligence (AI)', 'AI Engineer', 1),
(23, 'Artificial Intelligence (AI)', 'Data Scientist', 1),
(24, 'Artificial Intelligence (AI)', 'Data Analyst', 1),
(25, 'Artificial Intelligence (AI)', 'Machine Learning Engineer', 1),
(26, 'Artificial Intelligence (AI)', 'AI Researcher', 1),
(27, 'Artificial Intelligence (AI)', 'Natural Language Processing (NLP) Engineer', 1),
(28, 'Artificial Intelligence (AI)', 'Computer Vision Engineer', 1),
(29, 'Artificial Intelligence (AI)', 'Deep Learning Engineer', 1),
(30, 'Artificial Intelligence (AI)', 'AI Product Manager', 1),
(31, 'Cybersecurity', 'Cybersecurity Analyst', 1),
(32, 'Cybersecurity', 'Security Engineer', 1),
(33, 'Cybersecurity', 'Penetration Tester', 1),
(34, 'Cybersecurity', 'IT Security Specialist', 1),
(35, 'Cybersecurity', 'SOC Analyst', 1),
(36, 'Cybersecurity', 'Cryptographer', 1),
(37, 'Cybersecurity', 'Cloud Security Engineer', 1),
(38, 'Cybersecurity', 'Application Security Engineer', 1),
(39, 'Cybersecurity', 'Network Security Engineer', 1),
(40, 'Database Administration', 'Database Administrator', 1),
(41, 'Database Administration', 'Data Architect', 1),
(42, 'Database Administration', 'Database Developer', 1),
(43, 'Database Administration', 'DBA Support', 1),
(44, 'Database Administration', 'NoSQL Database Administrator', 1),
(45, 'Database Administration', 'Big Data Engineer', 1),
(46, 'Cloud Computing', 'Cloud Architect', 1),
(47, 'Cloud Computing', 'Cloud Engineer', 1),
(48, 'Cloud Computing', 'Cloud Solutions Architect', 1),
(49, 'Cloud Computing', 'Cloud Consultant', 1),
(50, 'Cloud Computing', 'Cloud Developer', 1),
(51, 'Cloud Computing', 'AWS Engineer', 1),
(52, 'Cloud Computing', 'Azure Engineer', 1),
(53, 'Cloud Computing', 'Google Cloud Engineer', 1),
(54, 'Cloud Computing', 'Cloud Security Engineer', 1),
(55, 'Network and Systems Administration', 'System Administrator', 1),
(56, 'Network and Systems Administration', 'Network Administrator', 1),
(57, 'Network and Systems Administration', 'Linux Administrator', 1),
(58, 'Network and Systems Administration', 'IT Support Engineer', 1),
(59, 'Network and Systems Administration', 'Cloud Systems Administrator', 1),
(60, 'Network and Systems Administration', 'Virtualization Engineer', 1),
(61, 'UI/UX Design', 'UI Designer', 1),
(62, 'UI/UX Design', 'UX Designer', 1),
(63, 'UI/UX Design', 'Interaction Designer', 1),
(64, 'UI/UX Design', 'Visual Designer', 1),
(65, 'UI/UX Design', 'Product Designer', 1),
(66, 'UI/UX Design', 'UX Researcher', 1),
(67, 'Game Development', 'Game Developer', 1),
(68, 'Game Development', 'Game Designer', 1),
(69, 'Game Development', 'Game Artist', 1),
(70, 'Game Development', 'Game Programmer', 1),
(71, 'Game Development', 'Game Producer', 1),
(72, 'Game Development', 'Game Animator', 1),
(73, 'Game Development', 'Game Sound Designer', 1),
(74, 'Blockchain Technology', 'Blockchain Developer', 1),
(75, 'Blockchain Technology', 'Blockchain Architect', 1),
(76, 'Blockchain Technology', 'Blockchain Consultant', 1),
(77, 'Blockchain Technology', 'Blockchain Researcher', 1),
(78, 'Blockchain Technology', 'Smart Contract Developer', 1),
(79, 'Technical Support', 'IT Support', 1),
(80, 'Technical Support', 'Technical Support Engineer', 1),
(81, 'Technical Support', 'Help Desk Technician', 1),
(82, 'Technical Support', 'Customer Support Engineer', 1),
(83, 'Technical Support', 'Product Support Engineer', 1),
(84, 'Business Analysis', 'Business Analyst', 1),
(85, 'Business Analysis', 'Data Analyst', 1),
(86, 'Business Analysis', 'Business Systems Analyst', 1),
(87, 'Business Analysis', 'Financial Analyst', 1),
(88, 'Business Analysis', 'Process Analyst', 1),
(89, 'Business Analysis', 'Product Analyst', 1),
(90, 'IT Project Management', 'IT Project Manager', 1),
(91, 'IT Project Management', 'Scrum Master', 1),
(92, 'IT Project Management', 'Agile Project Manager', 1),
(93, 'IT Project Management', 'Program Manager', 1),
(94, 'IT Project Management', 'Project Coordinator', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `chat`
--

CREATE TABLE `chat` (
  `id` int(11) NOT NULL,
  `sender_id` int(11) NOT NULL,
  `sender_role` text NOT NULL,
  `receiver_id` int(11) NOT NULL,
  `receiver_role` text NOT NULL,
  `message` text NOT NULL,
  `time` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `chat`
--

INSERT INTO `chat` (`id`, `sender_id`, `sender_role`, `receiver_id`, `receiver_role`, `message`, `time`, `status`) VALUES
(1, 65, 'Ứng viên', 1, 'Nhà tuyển dụng', 'Hi', '2025-03-25 12:27:28', 1),
(2, 65, 'SEEKER', 1, 'EMPLOYER', 'hello', '2025-03-25 12:27:40', 1),
(3, 65, 'SEEKER', 1, 'EMPLOYER', 'https://res.cloudinary.com/dw8kgfdq2/raw/upload/v1743226243/CV_NguyenHoangTu_Java%20Software%20Engineer.pdf', '2025-03-29 12:36:49', 1),
(4, 1, 'EMPLOYER', 65, 'SEEKER', 'hello', '2025-03-29 13:29:14', 1),
(14, 65, 'SEEKER', 1, 'EMPLOYER', 'Ứng viên đã ứng tuyển công việc <strong>PM/ Project Manager/ Brse (Japanese) - Yêu Cầu Tiếng Nhật N2 - 2 Năm Kinh Nghiệm</strong>. Đây là CV của Ứng viên: <a href=\"http://192.168.198.1:8081/assets/files/23b46d4b790b4f4e9a8908a41a718409.pdf\" target=\"_blank\">Xem CV</a>', '2025-03-29 16:55:40', 1),
(15, 65, 'SEEKER', 1, 'EMPLOYER', 'Ứng viên đã ứng tuyển công việc <strong>Ruby on Rails Developer</strong>. Đây là CV của Ứng viên: <a href=\"undefined\" target=\"_blank\">Xem CV</a>', '2025-04-12 21:10:53', 1),
(16, 65, 'SEEKER', 1, 'EMPLOYER', 'Ứng viên đã ứng tuyển công việc <strong>Ruby on Rails Developer</strong>. Đây là CV của Ứng viên: <a href=\"undefined\" target=\"_blank\">Xem CV</a>', '2025-04-12 21:15:20', 1),
(17, 65, 'SEEKER', 1, 'EMPLOYER', 'Ứng viên đã ứng tuyển công việc <strong>Ruby on Rails Developer</strong>. Đây là CV của Ứng viên: <a href=\"http://192.168.198.1:8081/assets/files/a42699d152fb46d3a5eca1027437355e.pdf\" target=\"_blank\">Xem CV</a>', '2025-04-12 21:20:49', 1),
(20, 1, 'EMPLOYER', 65, 'SEEKER', 'Bạn nhận được một bài kiểm tra cho vị trí <strong>Ruby on Rails Developer</strong> từ <strong>Công ty FireGroup</strong>. <br> Mã bài kiểm tra: <strong>java</strong>. <a href=\"http://localhost:4200/seeker/test/3\" target=\"_blank\"> <br>Bắt đầu bài kiểm tra</a>', '2025-04-27 15:08:54', 1),
(21, 1, 'EMPLOYER', 65, 'SEEKER', 'Bạn nhận được một bài kiểm tra cho vị trí <strong>Ruby on Rails Developer</strong> từ <strong>Công ty FireGroup</strong>. <br> Mã bài kiểm tra: <strong>java</strong>. <a href=\"http://localhost:4200/seeker/test/3\" target=\"_blank\"> <br>Bắt đầu bài kiểm tra</a>', '2025-04-27 15:35:59', 1),
(22, 1, 'EMPLOYER', 65, 'SEEKER', 'Bạn nhận được một bài kiểm tra cho vị trí <strong>Ruby on Rails Developer</strong> từ <strong>Công ty FireGroup</strong>. <br> Mã bài kiểm tra: <strong>java</strong>. <a href=\"http://localhost:4200/seeker/test/3\" target=\"_blank\"> <br>Bắt đầu bài kiểm tra</a>', '2025-04-27 16:00:41', 1),
(23, 1, 'EMPLOYER', 65, 'SEEKER', 'Bạn nhận được một bài kiểm tra cho vị trí <strong>Ruby on Rails Developer</strong> từ <strong>Công ty FireGroup</strong>. <br> Mã bài kiểm tra: <strong>java</strong>. <a href=\"http://localhost:4200/seeker/test/3\" target=\"_blank\"> <br>Bắt đầu bài kiểm tra</a>', '2025-04-27 16:03:11', 1),
(24, 95, 'SEEKER', 1, 'EMPLOYER', 'Ứng viên đã ứng tuyển công việc <strong>Fullstack Developer</strong>. Đây là CV của Ứng viên: <a href=\"http://192.168.198.1:8081/assets/files/12d79162228a4564aaa513cbf74c19e9.pdf\" target=\"_blank\">Xem CV</a>', '2025-06-01 00:59:19', 1),
(25, 1, 'EMPLOYER', 95, 'SEEKER', 'hello', '2025-06-09 21:27:02', 1),
(26, 1, 'EMPLOYER', 95, 'SEEKER', 'aa', '2025-06-09 21:28:40', 1),
(27, 1, 'EMPLOYER', 95, 'SEEKER', 'hel', '2025-06-09 21:30:12', 1),
(28, 1, 'EMPLOYER', 95, 'SEEKER', 'allll', '2025-06-09 21:38:10', 1),
(29, 1, 'EMPLOYER', 95, 'SEEKER', 'aa', '2025-06-09 21:41:45', 1),
(30, 1, 'EMPLOYER', 95, 'SEEKER', 'bb', '2025-06-09 21:44:15', 1),
(31, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 21:49:29', 0),
(32, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 21:51:24', 0),
(33, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 21:52:45', 0),
(34, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 21:57:28', 0),
(35, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 21:58:06', 0),
(36, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 21:58:12', 0),
(37, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 21:58:17', 0),
(38, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 21:58:19', 0),
(39, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 21:59:22', 0),
(40, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 22:00:39', 0),
(41, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 22:01:58', 0),
(42, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 22:03:33', 0),
(43, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 22:06:03', 0),
(44, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 22:06:38', 0),
(45, 1, 'EMPLOYER', 95, 'SEEKER', 'ccc', '2025-06-09 22:09:13', 1),
(46, 1, 'EMPLOYER', 95, 'SEEKER', 'ccccccaaaa', '2025-06-09 22:09:43', 1),
(47, 1, 'EMPLOYER', 95, 'SEEKER', 'vvv', '2025-06-09 22:10:09', 1),
(48, 1, 'EMPLOYER', 95, 'SEEKER', 'aav', '2025-06-09 22:16:07', 1),
(49, 1, 'EMPLOYER', 95, 'SEEKER', 'aaa', '2025-06-09 22:17:15', 1),
(50, 1, 'EMPLOYER', 95, 'SEEKER', 'ttt', '2025-06-09 22:34:33', 1),
(51, 1, 'EMPLOYER', 95, 'SEEKER', 'cc', '2025-06-09 22:41:40', 1),
(52, 1, 'EMPLOYER', 95, 'SEEKER', 'a', '2025-06-09 22:42:52', 1),
(53, 1, 'EMPLOYER', 95, 'SEEKER', 'anh iu em', '2025-06-09 22:47:47', 1),
(54, 1, 'EMPLOYER', 95, 'SEEKER', 'aaaa', '2025-06-09 22:48:00', 1),
(55, 1, 'EMPLOYER', 95, 'SEEKER', 'tình', '2025-06-09 22:52:03', 1),
(56, 1, 'EMPLOYER', 95, 'SEEKER', 'ccc', '2025-06-09 22:54:07', 1),
(57, 1, 'EMPLOYER', 95, 'SEEKER', 'aaa', '2025-06-09 22:54:21', 1),
(58, 1, 'EMPLOYER', 95, 'SEEKER', 'cccc', '2025-06-09 22:56:15', 1),
(59, 95, 'SEEKER', 1, 'EMPLOYER', 'tttt', '2025-06-09 22:57:10', 1),
(60, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 23:03:18', 0),
(61, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 23:05:12', 0),
(62, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 23:05:58', 0),
(63, 1, 'EMPLOYER', 95, 'SEEKER', 'ccc', '2025-06-09 23:07:46', 1),
(64, 1, 'EMPLOYER', 95, 'SEEKER', 'anh', '2025-06-09 23:10:41', 1),
(65, 1, 'EMPLOYER', 95, 'SEEKER', 'Tú', '2025-06-09 23:14:21', 1),
(66, 1, 'EMPLOYER', 95, 'SEEKER', 'cccccc', '2025-06-09 23:17:39', 1),
(67, 1, 'EMPLOYER', 95, 'SEEKER', 'Tú', '2025-06-09 23:19:00', 1),
(68, 1, 'EMPLOYER', 95, 'SEEKER', 'aaa', '2025-06-09 23:19:04', 1),
(69, 1, 'EMPLOYER', 95, 'SEEKER', 'ttt', '2025-06-09 23:19:42', 1),
(70, 1, 'EMPLOYER', 95, 'SEEKER', 'aaaa', '2025-06-09 23:22:06', 1),
(71, 95, 'SEEKER', 1, 'EMPLOYER', 'ccc', '2025-06-09 23:23:23', 1),
(72, 1, 'EMPLOYER', 95, 'SEEKER', 'aaa', '2025-06-09 23:23:28', 1),
(73, 1, 'EMPLOYER', 95, 'SEEKER', 'cccc', '2025-06-09 23:24:57', 1),
(74, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 23:28:46', 0),
(75, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 23:31:11', 0),
(76, 1, 'EMPLOYER', 95, 'SEEKER', 'bbbbb', '2025-06-09 23:34:12', 1),
(77, 1, 'EMPLOYER', 95, 'test', 'aaaa', '2025-06-09 23:37:10', 0),
(78, 1, 'EMPLOYER', 95, 'SEEKER', 'tttt', '2025-06-09 23:37:49', 1),
(79, 1, 'EMPLOYER', 65, 'SEEKER', 'chat', '2025-06-09 23:46:05', 1),
(80, 1, 'EMPLOYER', 65, 'SEEKER', 'ccc', '2025-06-09 23:48:50', 1),
(81, 1, 'EMPLOYER', 65, 'SEEKER', 'aaa', '2025-06-09 23:50:03', 1),
(82, 1, 'EMPLOYER', 95, 'SEEKER', 'tttt', '2025-06-09 23:52:07', 1),
(83, 1, 'EMPLOYER', 95, 'SEEKER', 'ttttt', '2025-06-09 23:52:52', 1),
(84, 1, 'EMPLOYER', 95, 'SEEKER', 'ccccc', '2025-06-09 23:53:04', 1),
(85, 1, 'EMPLOYER', 95, 'SEEKER', 'linh hôi nách', '2025-06-09 23:54:11', 1),
(86, 1, 'EMPLOYER', 95, 'SEEKER', 'aaaa', '2025-06-10 00:01:13', 1),
(87, 95, 'SEEKER', 1, 'EMPLOYER', 'vvv', '2025-06-10 00:01:28', 1),
(88, 1, 'EMPLOYER', 95, 'SEEKER', 'bbbbbb', '2025-06-10 00:01:37', 1),
(89, 1, 'EMPLOYER', 65, 'SEEKER', 'sao vậy', '2025-06-17 20:11:31', 1),
(90, 1, 'EMPLOYER', 95, 'SEEKER', 'cc', '2025-06-17 20:14:14', 1),
(91, 1, 'EMPLOYER', 95, 'SEEKER', 'tôi buồn', '2025-06-17 20:16:15', 1),
(92, 1, 'EMPLOYER', 95, 'SEEKER', 'chán quá', '2025-06-17 20:17:21', 1),
(93, 95, 'SEEKER', 1, 'EMPLOYER', 'sao chán', '2025-06-17 20:17:44', 1),
(94, 1, 'EMPLOYER', 95, 'SEEKER', 'cc', '2025-06-17 20:21:08', 1),
(95, 95, 'SEEKER', 1, 'EMPLOYER', 'hi', '2025-06-17 20:30:24', 1),
(96, 95, 'SEEKER', 1, 'EMPLOYER', 'huhu', '2025-06-17 20:34:26', 1),
(97, 95, 'SEEKER', 1, 'EMPLOYER', 'khỏe không', '2025-06-17 20:40:34', 1),
(98, 95, 'SEEKER', 1, 'EMPLOYER', 'ccc', '2025-06-17 20:43:41', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `cv`
--

CREATE TABLE `cv` (
  `id` int(11) NOT NULL,
  `seeker_id` int(11) NOT NULL,
  `name` text NOT NULL,
  `skills` text DEFAULT NULL,
  `experience` text DEFAULT NULL,
  `type` text DEFAULT NULL,
  `education` text DEFAULT NULL,
  `certification` text DEFAULT NULL,
  `status` tinyint(1) NOT NULL,
  `offer_salary` text DEFAULT NULL,
  `job_deadline` text DEFAULT NULL,
  `linked_in` text DEFAULT NULL,
  `link_git` text DEFAULT NULL,
  `upload_at` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `cv`
--

INSERT INTO `cv` (`id`, `seeker_id`, `name`, `skills`, `experience`, `type`, `education`, `certification`, `status`, `offer_salary`, `job_deadline`, `linked_in`, `link_git`, `upload_at`) VALUES
(2, 65, '53a44b67db694aca8d6c8a8a028a6ad1.pdf', '.net, angular, angularjs, apache, bootstrap, c, css, design patterns, docker, git, github, go, html, https, java, javascript, jquery, kafka, keycloak, laravel, less, microservices, mvc, mysql, oop, php, postgresql, python, r, react, reactjs, redis, rest api, spring, spring boot, sql server, tailwind css, wordpress', '[{\"company\":\"Yes4All Vietnam Company Limited\",\"position\":\"tu-nguyen-802b82234/  SOFTWARE ENGINEER INTERN\",\"desc\":\"\",\"time\":\"2/2025 - 3/2025\"}]', NULL, NULL, NULL, 1, NULL, NULL, NULL, NULL, '2025-05-08 20:55:06'),
(3, 95, '394f6b8090c040e39aa504082b9a3a7f.pdf', '.net, angular, angularjs, apache, bootstrap, c, css, design patterns, docker, git, github, go, html, https, java, javascript, jquery, kafka, keycloak, laravel, less, microservices, mvc, mysql, oop, php, postgresql, python, r, react, reactjs, redis, rest api, spring, spring boot, sql server, tailwind css, wordpress', '[{\"company\":\"Yes4All Vietnam Company Limited\",\"position\":\"tu-nguyen-802b82234/  SOFTWARE ENGINEER INTERN\",\"desc\":\"\",\"time\":\"2/2025 - 3/2025\"}]', NULL, NULL, NULL, 1, NULL, NULL, NULL, NULL, '2025-06-12 12:43:21');

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `employer`
--

CREATE TABLE `employer` (
  `id` int(11) NOT NULL,
  `company_name` text DEFAULT NULL,
  `company_profile` text DEFAULT NULL,
  `rating` double DEFAULT NULL,
  `contact_info` text DEFAULT NULL,
  `logo` text DEFAULT NULL,
  `cover_img` text DEFAULT NULL,
  `address` text DEFAULT NULL,
  `map_link` text DEFAULT NULL,
  `amount` double DEFAULT NULL,
  `status` tinyint(1) DEFAULT NULL,
  `description` text DEFAULT NULL,
  `founded_in` text DEFAULT NULL,
  `team_member` text DEFAULT NULL,
  `company_field` text DEFAULT NULL,
  `company_link` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `employer`
--

INSERT INTO `employer` (`id`, `company_name`, `company_profile`, `rating`, `contact_info`, `logo`, `cover_img`, `address`, `map_link`, `amount`, `status`, `description`, `founded_in`, `team_member`, `company_field`, `company_link`) VALUES
(1, 'Công ty FireGroup', 'FIREGROUP TECHNOLOGY\nFireGroup envisions and strives to make an e-Commerce world where every merchant can succeed a reality. Since our founding in 2016, we have specialized in providing a comprehensive portfolio of category-leading SaaS products for the global e-Commerce market, including online store development, store management, sales & marketing automation, and almost every aspect of running an online business.\n\nAfter almost a decade in the market, we are loved by over 450,000 merchants and enterprises across 175+ countries and highly trusted by our globally renowned partners such as Shopify, BigCommerce, WooCommerce, Google, Meta, Tiktok, and Amazon. To continue empowering more merchants, we always welcome new tech talents who work toward a rewarding career and live by these 4 core values:\n\nGrowth\nTrust\nEmbracing Change\nCustomer Centricity\nWhy to build your career with us\n\nMake your impact globally on hundreds of thousands of merchants’ success\nDevelop your skills in a fast-paced and highly competitive SaaS market\nCreate and grow continuously among passionate talents with growth mindsets', 0, 'acc1@gmail.com', '8dd661aefb2f484d85228df401cb1093.jpg', 'be8e26206e1f4f009c52daeed6b7bfce.jpg', 'TP Hồ Chí Minh - Đà Nẵng', '', 200000, 1, 'Không có OT', '2016', '300', 'Thương Mại Điện Tử', 'https://firegroup.io/'),
(2, 'FPT Software', 'FPT Software is a global IT company providing software development and IT services.', 0, 'contact@fptsoftware.com', '34538c77a91f4f4494596754d85dbec7.png', 'fbefed9351924f819b5f1f66256bcca0.png', 'Hà Nội', '', 5000000000, 1, 'One of the largest IT companies in Vietnam.', '1988', '48000', 'Công nghệ thông tin', 'https://fptsoftware.com'),
(3, 'Tiki Corporation', 'Tiki Corporation is a leading e-commerce platform in Vietnam.', 4.5, 'support@tiki.vn', NULL, NULL, 'TP Hồ Chí Minh', NULL, 2000000000, 1, 'Focuses on e-commerce and logistics.', '2010', '4000', 'Thương mại điện tử', 'https://tiki.vn'),
(4, 'Axon Active Vietnam', 'Axon Active Vietnam provides software development and IT outsourcing services.', 4.3, 'info@axonactive.com', NULL, NULL, 'TP Hồ Chí Minh', NULL, 1000000000, 1, 'Specializes in offshore software development.', '2008', '1000', 'Công nghệ thông tin', 'https://axonactive.com'),
(5, 'KMS Technology', 'KMS Technology focuses on software development and IT solutions.', 4.7, 'contact@kms-technology.com', NULL, NULL, 'TP Hồ Chí Minh', NULL, 1500000000, 1, 'Provides software development and testing services.', '2009', '1500', 'Công nghệ thông tin', 'https://kms-technology.com'),
(6, 'Haravan', 'Haravan provides e-commerce and retail solutions.', 4.2, 'support@haravan.com', NULL, NULL, 'TP Hồ Chí Minh', NULL, 500000000, 1, 'Specializes in e-commerce platforms and services.', '2014', '150', 'Thương mại điện tử', 'https://haravan.com'),
(7, 'TikiNow', 'TikiNow is a logistics service under Tiki Corporation.', 4, 'support@tiki.vn', NULL, NULL, 'TP Hồ Chí Minh', NULL, 300000000, 1, 'Focuses on fast delivery services.', '2016', '100', 'Logistics', 'https://tiki.vn'),
(8, 'Viettel Solutions', 'Viettel Solutions provides IT services and digital transformation solutions to businesses.', 4.6, 'contact@viettelsolutions.vn', 'https://viettelsolutions.vn/logo.png', 'https://viettelsolutions.vn/cover.jpg', 'Hà Nội', 'https://maps.google.com?viettel', 3000000000, 1, 'A leading IT arm of Viettel Group, focusing on digital innovation.', '2007', '2000', 'Công nghệ thông tin', 'https://viettelsolutions.vn'),
(9, 'TMA Solutions', 'TMA Solutions offers software outsourcing and development services globally.', 4.4, 'info@tmasolutions.com', 'https://tmasolutions.com/logo.png', 'https://tmasolutions.com/cover.jpg', 'TP Hồ Chí Minh', 'https://maps.google.com?tma', 1200000000, 1, 'A pioneer in software engineering with a strong global presence.', '1997', '3500', 'Công nghệ thông tin', 'https://tmasolutions.com'),
(10, 'Rikkei Soft', 'Rikkei Soft focuses on mobile and web development with innovative approaches.', 4.2, 'contact@rikkeisoft.com', 'https://rikkeisoft.com/logo.png', 'https://rikkeisoft.com/cover.jpg', 'Hà Nội', 'https://maps.google.com?rikkei', 500000000, 1, 'A growing IT firm specializing in mobile and web technologies.', '2012', '150', 'Công nghệ thông tin', 'https://rikkeisoft.com'),
(11, 'LogiGear Vietnam', 'LogiGear Vietnam provides software testing and quality assurance services.', 4.5, 'support@logigear.com', 'https://logigear.com/logo.png', 'https://logigear.com/cover.jpg', 'TP Hồ Chí Minh', 'https://maps.google.com?logigear', 800000000, 1, 'Known for its expertise in software quality assurance.', '2003', '800', 'Công nghệ thông tin', 'https://logigear.com'),
(12, 'Tek Experts Vietnam', 'Tek Experts offers tech support and IT services with global reach.', 4.3, 'info@tekexperts.com', 'https://tekexperts.com/logo.png', 'https://tekexperts.com/cover.jpg', 'Đà Nẵng', 'https://maps.google.com?tek', 700000000, 1, 'A global provider of technical support and IT solutions.', '2014', '400', 'Công nghệ thông tin', 'https://tekexperts.com'),
(13, 'Sun* Inc.', 'Sun* Inc. specializes in software development and artificial intelligence.', 4.7, 'contact@sun-asterisk.com', 'https://sun-asterisk.com/logo.png', 'https://sun-asterisk.com/cover.jpg', 'TP Hồ Chí Minh', 'https://maps.google.com?sun', 900000000, 1, 'An innovative firm focusing on AI and software solutions.', '2009', '1200', 'Công nghệ thông tin', 'https://sun-asterisk.com'),
(14, 'NashTech Vietnam', 'NashTech provides digital transformation and cloud services.', 4.4, 'info@nashtechglobal.com', 'https://nashtechglobal.com/logo.png', 'https://nashtechglobal.com/cover.jpg', 'Hà Nội', 'https://maps.google.com?nashtech', 1100000000, 1, 'A leader in digital and cloud transformation solutions.', '2000', '1500', 'Công nghệ thông tin', 'https://nashtechglobal.com'),
(15, 'FPT Telecom', 'FPT Telecom offers internet and telecom services across Vietnam.', 4.6, 'support@fpttelecom.vn', 'https://fpttelecom.vn/logo.png', 'https://fpttelecom.vn/cover.jpg', 'Hà Nội', 'https://maps.google.com?fpt', 2500000000, 1, 'A leading telecom provider with nationwide coverage.', '1997', '3000', 'Viễn thông', 'https://fpttelecom.vn'),
(16, 'CMC Global', 'CMC Global provides IT services and software solutions.', 4.5, 'contact@cmcglobal.com', 'https://cmcglobal.com/logo.png', 'https://cmcglobal.com/cover.jpg', 'TP Hồ Chí Minh', 'https://maps.google.com?cmc', 1300000000, 1, 'Part of CMC Corporation, focusing on IT innovation.', '2005', '1800', 'Công nghệ thông tin', 'https://cmcglobal.com'),
(17, 'SmartOSC', 'SmartOSC specializes in e-commerce and digital solutions.', 4.3, 'info@smartosc.com', 'https://smartosc.com/logo.png', 'https://smartosc.com/cover.jpg', 'Hà Nội', 'https://maps.google.com?smartosc', 600000000, 1, 'A leading e-commerce and digital agency in Vietnam.', '2006', '700', 'Thương mại điện tử', 'https://smartosc.com'),
(18, '10Clouds Vietnam', '10Clouds Vietnam offers software development and cloud solutions.', 4.2, 'contact@10clouds.com', 'https://10clouds.com/logo.png', 'https://10clouds.com/cover.jpg', 'Đà Nẵng', 'https://maps.google.com?10clouds', 400000000, 1, 'Focuses on cloud-based and mobile application development.', '2015', '200', 'Công nghệ thông tin', 'https://10clouds.com'),
(19, 'Sii Asia Vietnam', 'Sii Asia provides IT outsourcing and consulting services.', 4.4, 'info@siiasia.com', 'https://siiasia.com/logo.png', 'https://siiasia.com/cover.jpg', 'TP Hồ Chí Minh', 'https://maps.google.com?siiasia', 500000000, 1, 'Part of Sii Group, offering global IT services.', '2013', '300', 'Công nghệ thông tin', 'https://siiasia.com'),
(20, 'Digiworld', 'Digiworld distributes IT products and provides tech services.', 4.5, 'support@digiworld.com', 'https://digiworld.com/logo.png', 'https://digiworld.com/cover.jpg', 'Hà Nội', 'https://maps.google.com?digiworld', 800000000, 1, 'A leading IT distributor with a strong market presence.', '1996', '900', 'Phân phối công nghệ', 'https://digiworld.com'),
(21, 'VinAI', 'VinAI focuses on AI research and development for innovation.', 4.7, 'contact@vinai.io', 'https://vinai.io/logo.png', 'https://vinai.io/cover.jpg', 'Hà Nội', 'https://maps.google.com?vinai', 1000000000, 1, 'An AI research arm under Vingroup.', '2019', '150', 'Trí tuệ nhân tạo', 'https://vinai.io'),
(22, 'CyberLogitec Vietnam', 'CyberLogitec provides logistics software and solutions.', 4.3, 'info@cyberlogitec.com', 'https://cyberlogitec.com/logo.png', 'https://cyberlogitec.com/cover.jpg', 'TP Hồ Chí Minh', 'https://maps.google.com?cyberlogitec', 600000000, 1, 'Specializes in logistics and supply chain technology.', '2005', '500', 'Công nghệ logistics', 'https://cyberlogitec.com'),
(23, 'Global CyberSoft', 'Global CyberSoft offers IT services and software development.', 4.2, 'support@globalcybersoft.com', 'https://globalcybersoft.com/logo.png', 'https://globalcybersoft.com/cover.jpg', 'Đà Nẵng', 'https://maps.google.com?globalcybersoft', 400000000, 1, 'A mid-sized IT firm with growing potential.', '2011', '180', 'Công nghệ thông tin', 'https://globalcybersoft.com'),
(24, 'Halotech', 'Halotech provides software solutions and IT support.', 4.1, 'contact@halotech.vn', 'https://halotech.vn/logo.png', 'https://halotech.vn/cover.jpg', 'Hà Nội', 'https://maps.google.com?halotech', 300000000, 1, 'A small IT startup with innovative projects.', '2018', '80', 'Công nghệ thông tin', 'https://halotech.vn'),
(25, 'KiotViet', 'KiotViet offers POS and e-commerce management solutions.', 4.3, 'info@kiotviet.vn', 'https://kiotviet.vn/logo.png', 'https://kiotviet.vn/cover.jpg', 'TP Hồ Chí Minh', 'https://maps.google.com?kiotviet', 450000000, 1, 'A leader in retail management software in Vietnam.', '2013', '120', 'Thương mại điện tử', 'https://kiotviet.vn'),
(26, 'VCCorp', 'VCCorp provides digital marketing and technology services.', 4.6, 'support@vccorp.vn', 'https://vccorp.vn/logo.png', 'https://vccorp.vn/cover.jpg', 'Hà Nội', 'https://maps.google.com?vccorp', 900000000, 1, 'A leading digital media and technology company.', '2007', '1100', 'Truyền thông số', 'https://vccorp.vn'),
(27, 'G-Group', 'G-Group offers IT training and software development services.', 4.2, 'contact@ggroup.vn', 'https://ggroup.vn/logo.png', 'https://ggroup.vn/cover.jpg', 'Đà Nẵng', 'https://maps.google.com?ggroup', 350000000, 1, 'Focuses on IT education and technology training.', '2016', '90', 'Đào tạo công nghệ', 'https://ggroup.vn'),
(88, 'Công ty Yes4all', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1, NULL, NULL, NULL, NULL, NULL);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `employermembership`
--

CREATE TABLE `employermembership` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `membership_id` int(11) NOT NULL,
  `start_date` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `end_date` datetime NOT NULL,
  `renewal_date` datetime NOT NULL,
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `employermembership`
--

INSERT INTO `employermembership` (`id`, `user_id`, `membership_id`, `start_date`, `end_date`, `renewal_date`, `status`) VALUES
(1, 65, 3, '2025-05-29 21:22:35', '2025-05-10 10:05:46', '2025-05-10 10:05:46', 0),
(2, 1, 4, '2025-05-29 21:51:55', '2025-06-29 16:35:42', '2025-05-29 16:35:42', 1),
(4, 95, 4, '2025-06-01 01:26:08', '2026-06-01 01:26:08', '2026-06-01 01:26:08', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `experience`
--

CREATE TABLE `experience` (
  `id` int(11) NOT NULL,
  `name` text NOT NULL,
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `experience`
--

INSERT INTO `experience` (`id`, `name`, `status`) VALUES
(1, 'Sắp đi làm', 1),
(2, 'Dưới 1 năm', 1),
(3, '1 năm', 1),
(4, '2 năm', 1),
(5, '3 năm', 1),
(6, '4 năm', 1),
(7, '5 năm', 1),
(8, 'Trên 5 năm', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `favorite`
--

CREATE TABLE `favorite` (
  `id` int(11) NOT NULL,
  `seeker_id` int(11) NOT NULL,
  `job_id` int(11) NOT NULL,
  `status` tinyint(1) NOT NULL,
  `created` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `favorite`
--

INSERT INTO `favorite` (`id`, `seeker_id`, `job_id`, `status`, `created`) VALUES
(2, 65, 101, 1, '2025-04-04'),
(5, 65, 102, 1, '2025-04-10'),
(20, 95, 140, 1, '2025-05-31');

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `feedback`
--

CREATE TABLE `feedback` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `content` text NOT NULL,
  `created_at` datetime NOT NULL,
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `feedback`
--

INSERT INTO `feedback` (`id`, `user_id`, `content`, `created_at`, `status`) VALUES
(1, 65, 'User: Hoàng Tú Nguyễn - Email: tuhoangnguyen2003@gmail.com - Feedback: aaa', '2025-04-06 14:26:43', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `follow`
--

CREATE TABLE `follow` (
  `id` int(11) NOT NULL,
  `employer_id` int(11) NOT NULL,
  `seeker_id` int(11) NOT NULL,
  `status` tinyint(1) NOT NULL,
  `created` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `follow`
--

INSERT INTO `follow` (`id`, `employer_id`, `seeker_id`, `status`, `created`) VALUES
(17, 1, 65, 1, '2025-04-18'),
(18, 1, 87, 1, '2025-04-22'),
(19, 88, 65, 0, '2025-04-23'),
(20, 1, 95, 1, '2025-06-03');

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `image`
--

CREATE TABLE `image` (
  `id` int(11) NOT NULL,
  `name` text NOT NULL,
  `user_id` int(11) NOT NULL,
  `user_type` int(11) NOT NULL,
  `status` tinyint(1) NOT NULL,
  `created` date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `interview`
--

CREATE TABLE `interview` (
  `id` int(11) NOT NULL,
  `application_id` int(11) NOT NULL,
  `scheduled_at` datetime NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `interview_link` text NOT NULL,
  `status` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `interview`
--

INSERT INTO `interview` (`id`, `application_id`, `scheduled_at`, `interview_link`, `status`) VALUES
(25, 90, '2025-04-27 22:00:00', 'https://meet.google.com/kpz-vhhn-yez', 2),
(26, 90, '2025-05-06 05:04:00', 'https://meet.google.com/pia-dnqh-cqs', 1),
(28, 91, '2025-05-25 16:00:00', 'link', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `job`
--

CREATE TABLE `job` (
  `id` int(11) NOT NULL,
  `employer_id` int(11) NOT NULL,
  `title` text NOT NULL,
  `description` text DEFAULT NULL,
  `description_json` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `required` text NOT NULL,
  `address` text NOT NULL,
  `location_id` int(11) NOT NULL,
  `salary` text NOT NULL,
  `status` tinyint(1) NOT NULL,
  `posted_at` datetime NOT NULL,
  `posted_expired` datetime NOT NULL,
  `experience_id` int(11) NOT NULL,
  `required_skills` text NOT NULL,
  `member` text NOT NULL,
  `work_type_id` int(11) NOT NULL,
  `category_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `job`
--

INSERT INTO `job` (`id`, `employer_id`, `title`, `description`, `description_json`, `required`, `address`, `location_id`, `salary`, `status`, `posted_at`, `posted_expired`, `experience_id`, `required_skills`, `member`, `work_type_id`, `category_id`) VALUES
(31, 1, 'Nodejs Developer', 'We are seeking a skilled Node.js Developer to join our dynamic engineering team in Hanoi. You’ll design, develop, and maintain robust backend systems, focusing on creating scalable RESTful APIs using Node.js and Express. Your responsibilities include integrating MongoDB databases, optimizing server-side performance, and ensuring system security through best practices like JWT authentication and input validation. You’ll collaborate with frontend developers to deliver seamless integrations and participate in code reviews to maintain high-quality standards. Expect to work in an agile environment with tools like Git, Docker, and Postman, contributing to innovative projects that impact thousands of users.', '{\"job_description\":{\"overview\":\"We are seeking a skilled Node.js Developer to join our dynamic engineering team in Hanoi. You will play a key role in designing, developing, and maintaining robust backend systems, focusing on creating scalable RESTful APIs using Node.js and Express to support innovative projects impacting thousands of users.\",\"responsibilities\":[\"Design and develop scalable RESTful APIs using Node.js and Express.\",\"Integrate and manage MongoDB databases, ensuring efficient schema design, indexing, and aggregation pipelines.\",\"Optimize server-side performance through techniques like caching with Redis and asynchronous programming.\",\"Ensure system security by implementing best practices such as JWT authentication, input validation, and OWASP Top 10 mitigation.\",\"Collaborate with frontend developers to ensure seamless API integrations.\",\"Participate in code reviews to maintain high-quality standards and contribute to an agile development environment using tools like Git, Docker, and Postman.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 3 months\' salary.\",\"additional_benefits\":[\"Comprehensive health insurance and social benefits as per Vietnamese law.\",\"Annual team-building trips and company events.\",\"Opportunities for training and certification in Node.js and related technologies.\",\"Modern workspace with free snacks and coffee in Hanoi.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'Node.js Developer Application\'.\"}', 'Minimum 3 years of experience with Node.js and Express in production environments. Proficiency in MongoDB (schema design, indexing, and aggregation pipelines) and RESTful API development (including versioning and documentation with Swagger). Strong knowledge of asynchronous programming, error handling, and performance optimization (e.g., caching with Redis). Familiarity with security practices (e.g., OWASP Top 10 mitigation) and testing frameworks (e.g., Mocha, Jest). Excellent problem-solving skills and experience with Git workflows.', 'Hà Nội', 1, '15000000', 0, '2025-05-22 12:56:21', '2025-07-01 00:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(82, 88, 'Backend Software Engineer', 'We are looking for a skilled Backend Software Engineer to join our team. In this role, you will design, implement, and maintain robust backend systems for our applications. You will work with APIs, databases, and third-party services. You will be expected to optimize system performance and ensure scalability. You will work with technologies like Node.js, Express, and MongoDB. The ideal candidate will have experience in server-side programming, API integration, and handling large-scale systems in production environments.', '{\n  \"job_description\": {\n    \"overview\": \"We are seeking a talented Backend Software Engineer to join our team in Hanoi. You will design, implement, and maintain robust backend systems for our applications, ensuring high performance and scalability while working with cutting-edge technologies like Node.js, Express, and MongoDB.\",\n    \"responsibilities\": [\n      \"Design and implement robust backend systems using Node.js and Express.\",\n      \"Build and maintain RESTful APIs for seamless integration with frontend applications.\",\n      \"Work with databases like MongoDB and PostgreSQL to manage and optimize data storage and retrieval.\",\n      \"Integrate third-party services and APIs to enhance application functionality.\",\n      \"Optimize system performance and ensure scalability for large-scale production environments.\",\n      \"Collaborate with cross-functional teams to deliver high-quality software solutions.\"\n    ]\n  },\n  \"benefits\": {\n    \"salary_range\": \"17,000,000 VND/month (net)\",\n    \"bonus\": \"Performance-based bonus, up to 2 months\' salary.\",\n    \"additional_benefits\": [\n      \"Health and social insurance as per Vietnamese regulations.\",\n      \"Annual company retreat and team-building activities.\",\n      \"Access to cloud certification programs (AWS, Azure).\",\n      \"Flexible remote work options after probation period.\"\n    ]\n  },\n  \"work_hours\": \"Monday to Friday, 8:30 AM - 5:30 PM\",\n  \"application_method\": \"Apply online via our career portal at www.company.com/careers with your CV and a brief introduction.\"\n}', 'Minimum 4 years of experience with Node.js and Express. Strong experience with databases (MongoDB, PostgreSQL). Experience in building RESTful APIs and optimizing server-side performance. Knowledge of cloud platforms like AWS or Azure is a plus.', 'Hà Nội', 1, '17000000', 1, '2025-04-11 00:00:00', '2025-05-12 00:00:00', 4, 'Node.js, Express, MongoDB, PostgreSQL, RESTful API', '1', 1, 1),
(83, 1, 'UI Developer', 'As a UI Developer, you will focus on building intuitive and beautiful user interfaces using React.js. You will work with designers to create seamless user experiences and collaborate with backend developers to integrate APIs. Your tasks will include improving frontend performance, ensuring cross-browser compatibility, and implementing responsive web designs. If you enjoy working with cutting-edge frontend technologies and have a keen eye for design, we want you on our team!', '{\n  \"job_description\": {\n    \"overview\": \"We are looking for a passionate UI Developer to join our frontend team in Ho Chi Minh City. You will focus on building intuitive and visually stunning user interfaces using React.js, collaborating with designers and backend developers to deliver seamless user experiences.\",\n    \"responsibilities\": [\n      \"Develop intuitive and beautiful user interfaces using React.js, JavaScript, CSS, and HTML.\",\n      \"Collaborate with designers to transform wireframes and mockups into responsive web designs.\",\n      \"Integrate RESTful APIs with backend developers to ensure smooth data flow.\",\n      \"Optimize frontend performance for faster load times and better user experience.\",\n      \"Ensure cross-browser compatibility and implement mobile-first design principles.\",\n      \"Use Git for version control and participate in code reviews to maintain code quality.\"\n    ]\n  },\n  \"benefits\": {\n    \"salary_range\": \"13,000,000 VND/month (net)\",\n    \"bonus\": \"Performance-based bonus, up to 1 month\'s salary.\",\n    \"additional_benefits\": [\n      \"Health insurance and social benefits as per Vietnamese law.\",\n      \"Yearly team outings and company events in Ho Chi Minh City.\",\n      \"Free access to online learning platforms for frontend development skills.\",\n      \"Modern office with a creative and collaborative environment.\"\n    ]\n  },\n  \"work_hours\": \"Monday to Friday, 9:00 AM - 6:00 PM\",\n  \"application_method\": \"Submit your CV and portfolio to hr@company.com with the subject \'UI Developer Application\'.\"\n}', 'Proficiency in React.js, JavaScript, CSS, HTML, and experience with frontend optimization. Experience in responsive design and mobile-first approaches. Familiarity with RESTful APIs and version control tools like Git.', 'Hồ Chí Minh', 2, '13000000', 1, '2025-04-12 00:00:00', '2025-07-12 00:00:00', 3, 'React.js, JavaScript, CSS, HTML, Frontend Development', '1', 1, 1),
(84, 1, 'Cloud Solutions Architect', 'We are seeking an experienced Cloud Solutions Architect to design and implement scalable cloud solutions for our applications. You will be responsible for creating cloud infrastructure, setting up automated deployment pipelines, and ensuring high availability and fault tolerance of our systems. You will work with cloud platforms like AWS, Google Cloud, and Azure. The ideal candidate should have a deep understanding of cloud security, cost optimization, and best practices for cloud architecture.', NULL, 'Experience with AWS, Google Cloud, Azure, and cloud infrastructure design. Strong understanding of cloud security, cost optimization, and deployment strategies. Ability to collaborate with development and operations teams to implement cloud solutions.', 'Đà Nẵng', 3, '20000000', 1, '2025-04-13 00:00:00', '2025-07-13 00:00:00', 6, 'AWS, Google Cloud, Azure, Cloud Architecture, DevOps', '1', 1, 6),
(85, 1, 'Frontend Engineer', 'Join our team as a Frontend Engineer, where you will be building dynamic, user-friendly web applications. You will be working with the design team to implement creative UI/UX features, focusing on performance, scalability, and usability. You will work with technologies such as React, HTML5, CSS3, and JavaScript. You should have a deep understanding of web performance and be able to implement optimizations and debugging for smooth user experience.', NULL, 'Experience with React.js, HTML5, CSS3, JavaScript, and frontend performance optimization. Strong understanding of web standards, cross-browser compatibility, and testing/debugging tools. Familiarity with version control systems like Git.', 'Hà Nội', 1, '12000000', 1, '2025-04-14 00:00:00', '2025-07-14 00:00:00', 2, 'React.js, JavaScript, HTML, CSS, Web Development', '1', 1, 1),
(86, 1, 'Mobile App Developer', 'We are looking for a Mobile App Developer to create high-quality applications for Android and iOS. You will be responsible for the development, testing, and deployment of mobile applications. You will use Flutter to create cross-platform applications and work with APIs to integrate data into the app. The ideal candidate should have experience building mobile applications and be comfortable working in an agile environment.', NULL, 'Proven experience with Flutter, Dart, and mobile app development. Knowledge of Android and iOS application lifecycle. Familiarity with mobile backend integration using APIs and cloud storage solutions.', 'Hồ Chí Minh', 2, '14000000', 1, '2025-04-15 00:00:00', '2025-07-15 00:00:00', 3, 'Flutter, Dart, Firebase, Mobile Development, API Integration', '1', 1, 1),
(87, 1, 'Security Engineer', 'As a Security Engineer, you will be responsible for securing our applications, infrastructure, and data. You will conduct security assessments, vulnerability scans, and threat analysis to identify and mitigate risks. You will also be tasked with ensuring compliance with security standards and best practices. The ideal candidate should have expertise in network security, cryptography, and ethical hacking practices.', NULL, 'Experience with security tools like Kali Linux, Metasploit, and Wireshark. Strong knowledge of network security protocols, cryptography, and vulnerability assessment techniques. Experience with cloud security and compliance standards like GDPR, HIPAA.', 'Đà Nẵng', 3, '16000000', 1, '2025-04-16 00:00:00', '2025-07-16 00:00:00', 2, 'Network Security, Ethical Hacking, Cryptography, Risk Assessment', '1', 4, 4),
(88, 1, 'Data Engineer', 'We are hiring a Data Engineer to help build and manage our data pipelines. You will be responsible for creating systems that collect, process, and analyze large datasets. You will work with big data tools such as Hadoop, Spark, and Kafka to ensure data is processed in real-time. The ideal candidate should have experience with data warehousing, ETL pipelines, and cloud-based data storage.', NULL, 'Experience with big data technologies like Hadoop, Spark, Kafka, and ETL processes. Proficiency in SQL and NoSQL databases. Ability to design and manage scalable data pipelines and work with cloud storage services like AWS S3 or Google Cloud Storage.', 'Hà Nội', 1, '17000000', 1, '2025-04-17 00:00:00', '2025-07-17 00:00:00', 4, 'Hadoop, Spark, Kafka, SQL, Data Warehousing', '1', 3, 5),
(89, 1, 'Artificial Intelligence Engineer', 'Join our team as an AI Engineer and work on cutting-edge artificial intelligence projects. You will build and optimize machine learning models for a variety of applications, such as computer vision, natural language processing, and recommendation systems. You will work with frameworks like TensorFlow, PyTorch, and Keras to create intelligent systems that learn and improve over time.', NULL, 'Strong knowledge of machine learning algorithms, deep learning, and AI frameworks like TensorFlow, Keras, and PyTorch. Experience with Python and data preprocessing. Ability to design, train, and deploy AI models for real-world applications.', 'Hồ Chí Minh', 2, '18000000', 1, '2025-04-18 00:00:00', '2025-07-18 00:00:00', 5, 'AI, TensorFlow, Keras, Deep Learning, Python', '1', 3, 3),
(90, 1, 'Blockchain Developer', 'As a Blockchain Developer, you will be responsible for developing blockchain-based applications, including smart contracts, dApps, and decentralized solutions. You will design and implement secure blockchain protocols and integrate them with backend systems. The ideal candidate will have a solid understanding of blockchain technology, cryptographic techniques, and decentralized systems.', NULL, 'Experience with blockchain platforms like Ethereum, Hyperledger, or Solana. Proficiency in Solidity and smart contract development. Understanding of consensus algorithms, cryptographic methods, and decentralized application (dApp) development.', 'Hà Nội', 1, '19000000', 1, '2025-04-19 00:00:00', '2025-07-19 00:00:00', 6, 'Blockchain, Solidity, Ethereum, Hyperledger, dApps', '1', 4, 7),
(91, 1, 'Game Developer', 'We are looking for a Game Developer to join our studio and help create immersive game experiences. You will work with Unity or Unreal Engine to develop game features, mechanics, and levels. You will collaborate with artists, designers, and other developers to create fun and engaging games. The ideal candidate should have experience with game engines, game physics, and multiplayer systems.', NULL, 'Experience with Unity or Unreal Engine. Knowledge of game physics, AI for games, and multiplayer systems. Strong C# or C++ programming skills. Experience in game design, gameplay mechanics, and user interface integration.', 'Hồ Chí Minh', 2, '16000000', 1, '2025-04-20 00:00:00', '2025-07-20 00:00:00', 2, 'Unity, Unreal Engine, C#, C++, Game Development', '1', 4, 8),
(92, 1, 'Java Developer', 'We are looking for a Java Developer to join our team. You will be responsible for building scalable backend systems using Java and Spring Boot. You will work closely with other developers to ensure the application meets performance and reliability standards. Your tasks will include creating RESTful APIs, optimizing the database, and ensuring secure and efficient data handling. If you are passionate about backend development and have a strong background in Java, we want to hear from you.', '{\n  \"job_description\": {\n    \"overview\": \"We are seeking a dedicated Java Developer to join our team in Hanoi. You will be responsible for building scalable backend systems using Java and Spring Boot, ensuring high performance and reliability while working on innovative projects.\",\n    \"responsibilities\": [\n      \"Develop scalable backend systems using Java and Spring Boot.\",\n      \"Design and implement RESTful APIs to support frontend applications.\",\n      \"Optimize database performance using SQL and Hibernate for efficient data handling.\",\n      \"Ensure application security by implementing best practices and secure coding techniques.\",\n      \"Collaborate with other developers to meet performance and reliability standards.\",\n      \"Participate in code reviews and contribute to improving development processes.\"\n    ]\n  },\n  \"benefits\": {\n    \"salary_range\": \"16,000,000 VND/month (net)\",\n    \"bonus\": \"Performance-based bonus, up to 2 months\' salary.\",\n    \"additional_benefits\": [\n      \"Comprehensive health and social insurance as per Vietnamese law.\",\n      \"Annual company trip and team-building events.\",\n      \"Support for Java and Spring certification programs.\",\n      \"Modern workspace in Hanoi with a focus on collaboration.\"\n    ]\n  },\n  \"work_hours\": \"Monday to Friday, 8:30 AM - 5:30 PM\",\n  \"application_method\": \"Apply online via our career portal at www.company.com/careers with your CV and a brief introduction.\"\n}', 'Experience with Java, Spring Boot, Hibernate, and RESTful APIs. Proficiency in SQL and database optimization. Strong knowledge of application security and performance optimization techniques.', 'Hà Nội', 1, '16000000', 1, '2025-04-21 00:00:00', '2025-07-21 00:00:00', 4, 'Java, Spring Boot, Hibernate, RESTful API', '1', 1, 1),
(93, 1, 'C++ Software Engineer', 'Join our team as a C++ Software Engineer! You will work on developing high-performance applications, focusing on system-level programming, algorithms, and data structures. You will contribute to the design and optimization of real-time applications and help improve the performance of complex software solutions. The ideal candidate will have experience with C++ development, understanding memory management, and working with low-latency systems.', NULL, 'Experience with C++ programming and system-level development. Strong understanding of algorithms, data structures, and memory management. Experience with performance optimization and debugging tools.', 'Hồ Chí Minh', 2, '17000000', 1, '2025-04-22 00:00:00', '2025-07-22 00:00:00', 5, 'C++, Algorithms, System-level Programming, Real-time Applications', '1', 1, 1),
(94, 1, 'PHP Developer', 'We are seeking a PHP Developer to work on developing server-side web applications. You will work with various PHP frameworks such as Laravel and Symfony to build scalable and secure web solutions. Your responsibilities will include building new features, optimizing database queries, and maintaining backend systems. You should have experience in both object-oriented programming and modern web development practices.', NULL, 'Experience with PHP, Laravel, Symfony, MySQL, and RESTful APIs. Strong understanding of web security best practices and performance optimization.', 'Đà Nẵng', 3, '15000000', 1, '2025-04-23 00:00:00', '2025-07-23 00:00:00', 3, 'PHP, Laravel, Symfony, MySQL, RESTful API', '1', 1, 1),
(95, 1, '.NET Developer', 'We are looking for a .NET Developer to build and maintain high-quality applications using .NET technologies. You will work on both web and desktop applications, collaborating with a team of developers to deliver functional and robust software. Your role will involve working with C#, ASP.NET Core, and SQL Server to build scalable systems and APIs. You should have strong problem-solving skills and the ability to optimize software performance.', NULL, 'Experience with C#, ASP.NET Core, SQL Server, and building web APIs. Knowledge of OOP principles and design patterns. Strong debugging and performance optimization skills.', 'Hà Nội', 1, '14500000', 1, '2025-04-24 00:00:00', '2025-07-24 00:00:00', 4, '.NET, C#, ASP.NET Core, SQL Server', '1', 1, 1),
(96, 1, 'Python Developer', 'Join us as a Python Developer and be responsible for designing and developing backend applications. You will work with frameworks such as Django and Flask to build robust APIs and services. Your tasks will include database management, working with cloud platforms, and implementing machine learning models. If you have a passion for Python and enjoy working in an agile environment, this is the job for you.', NULL, 'Experience with Python, Django, Flask, and PostgreSQL. Familiarity with cloud platforms like AWS and Google Cloud. Knowledge of machine learning libraries is a plus.', 'Hồ Chí Minh', 2, '16000000', 1, '2025-04-25 00:00:00', '2025-07-25 00:00:00', 3, 'Python, Django, Flask, PostgreSQL, AWS', '1', 1, 1),
(97, 1, 'React.js Developer', 'We are looking for a skilled React.js Developer to join our frontend team. You will be responsible for building interactive and dynamic user interfaces, focusing on performance, accessibility, and cross-browser compatibility. You will work closely with the design and backend teams to ensure seamless integration of frontend and backend services. If you are passionate about creating great user experiences and are familiar with the latest frontend technologies, we want you on our team.', NULL, 'Experience with React.js, Redux, JavaScript, HTML5, and CSS3. Strong understanding of web performance optimization and testing. Familiarity with RESTful APIs and working with backend teams.', 'Đà Nẵng', 3, '14000000', 1, '2025-04-26 00:00:00', '2025-07-26 00:00:00', 2, 'React.js, JavaScript, Redux, HTML5, CSS3, Web Development', '1', 1, 1),
(98, 1, 'Mobile Software Developer', 'We are hiring a Mobile Software Developer to create cutting-edge mobile applications. You will be responsible for designing and developing Android and iOS apps using Flutter and Dart. Your tasks will include integrating mobile applications with APIs, improving app performance, and ensuring a seamless user experience. You will also work with the team to develop features and ensure that the applications meet both user and business needs.', NULL, 'Experience with Flutter, Dart, Firebase, and mobile app development. Familiarity with RESTful APIs and third-party mobile integrations.', 'Hà Nội', 1, '15500000', 1, '2025-04-27 00:00:00', '2025-07-27 00:00:00', 3, 'Flutter, Dart, Firebase, Mobile Development, API Integration', '1', 1, 1),
(99, 1, 'Game Developer', 'We are looking for a talented Game Developer to help create immersive and engaging games. You will be responsible for building the game mechanics, designing levels, and creating interactive experiences for players. You will work with game engines like Unity or Unreal Engine to develop games for PC, console, and mobile platforms. The ideal candidate should have strong C# or C++ skills and be familiar with game development principles.', NULL, 'Experience with Unity or Unreal Engine, C# or C++. Knowledge of game physics, AI for games, and multiplayer systems. Ability to develop game mechanics and levels for a variety of game genres.', 'Hồ Chí Minh', 2, '18000000', 1, '2025-04-28 00:00:00', '2025-07-28 00:00:00', 4, 'Unity, Unreal Engine, C#, C++, Game Development', '1', 1, 8),
(100, 1, 'Automation Tester', 'We are looking for an experienced Automation Tester to join our QA team. You will be responsible for automating test cases, performing regression testing, and ensuring that our software meets the highest quality standards. You will work with testing frameworks like Selenium and Appium, and collaborate with the development team to integrate tests into the CI/CD pipeline. The ideal candidate will have experience in test automation and knowledge of software testing best practices.', NULL, 'Experience with test automation tools like Selenium, Appium, and JUnit. Strong knowledge of CI/CD pipelines and integration testing. Experience with web and mobile application testing.', 'Đà Nẵng', 3, '14500000', 1, '2025-04-29 00:00:00', '2025-07-29 00:00:00', 2, 'Selenium, Appium, JUnit, CI/CD, Test Automation', '1', 2, 2),
(101, 1, 'Ruby on Rails Developer', 'Join our team as a Ruby on Rails Developer! You will work on building web applications using the Ruby on Rails framework. Your primary responsibility will be creating server-side logic and ensuring high performance and scalability. You will work closely with frontend developers to deliver seamless integrations and contribute to code quality through reviews and testing. If you are passionate about Ruby and web development, this is the job for you.', NULL, 'Experience with Ruby on Rails, SQL, and RESTful API development. Strong understanding of MVC architecture, performance optimization, and best practices for building scalable web applications.', 'Hà Nội', 1, '15000000', 1, '2025-04-30 00:00:00', '2025-07-30 00:00:00', 4, 'Ruby on Rails, SQL, RESTful API, Web Development', '1', 1, 1),
(102, 1, 'Lập trình viên Java', '<p>Lập trình viên Java</p>', '{\n  \"job_description\": {\n    \"overview\": \"We are looking for an experienced Java Developer to join our team in Ho Chi Minh City. You will be responsible for developing high-performance backend applications using Java and Spring, leveraging modern technologies like AWS, Docker, and CI/CD pipelines to deliver scalable solutions.\",\n    \"responsibilities\": [\n      \"Develop and maintain backend applications using Java and Spring Framework.\",\n      \"Design and implement RESTful APIs to support various business requirements.\",\n      \"Manage and optimize PostgreSQL databases for efficient data storage and retrieval.\",\n      \"Deploy applications on AWS using Docker containers and CI/CD pipelines.\",\n      \"Collaborate with cross-functional teams to ensure seamless integration and delivery.\",\n      \"Participate in code reviews and ensure adherence to coding standards and best practices.\"\n    ]\n  },\n  \"benefits\": {\n    \"salary_range\": \"50,000,000 VND/month (net)\",\n    \"bonus\": \"Performance-based bonus, up to 4 months\' salary.\",\n    \"additional_benefits\": [\n      \"Premium health insurance and social benefits as per Vietnamese law.\",\n      \"Annual international company trip and team-building activities.\",\n      \"Support for AWS and Docker certification programs.\",\n      \"Flexible work-from-home options after probation.\"\n    ]\n  },\n  \"work_hours\": \"Monday to Friday, 9:00 AM - 6:00 PM\",\n  \"application_method\": \"Submit your CV and cover letter via email to hr@company.com with the subject \'Java Developer Application - HCMC\'.\"\n}', 'Java, Spring, SQL, Postgresql, AWS, Docker, CI/CD', 'Thành phố Hồ Chí Minh', 2, '50000000', 1, '2025-04-03 12:07:13', '2025-04-24 00:00:00', 4, 'Java, Spring, SQL, Postgresql, AWS, Docker, CI/CD', '3', 1, 1),
(110, 1, 'Software Engineer', NULL, '{\"job_description\":{\"overview\":\"As a Software Engineer, you will be responsible for the development and maintenance of scalable web applications. You will work closely with product managers and designers to deliver high-quality products.\",\"responsibilities\":[\"Design, develop, and maintain backend and frontend applications.\",\"Collaborate with cross-functional teams to define, design, and ship new features.\"]},\"benefits\":{\"salary_range\":\"\",\"bonus\":\"13th month salary and annual performance-based bonus.\",\"additional_benefits\":[\"Premium healthcare insurance.\",\"Flexible working hours and remote work options.\"]},\"work_hours\":\"8:00 - 17:00\",\"application_method\":\"\"}', 'Có kinh nghiệm 2 năm về Software', 'Tòa nhà Bitesco Quận 1', 2, '15,000,000 / tháng (Gross)', 1, '2025-04-13 16:38:10', '2025-04-30 00:00:00', 1, 'Java, SpringBoot, Mysql, PostgreSQL, Docker, CI/CD', '1', 1, 1),
(112, 1, 'Java Backend Engineer', NULL, '{\"job_description\":{\"overview\":\"As a Software Engineer, you will be responsible for the development and maintenance of scalable web applications. You will work closely with product managers and designers to deliver high-quality products.\",\"responsibilities\":[\"Design, develop, and maintain backend and frontend applications.\",\"Collaborate with cross-functional teams to define, design, and ship new features.\"]},\"benefits\":{\"salary_range\":\"\",\"bonus\":\"13th month salary and annual performance-based bonus.\",\"additional_benefits\":[\"Premium healthcare insurance.\",\"Flexible working hours and remote work options.\"]},\"work_hours\":\"8:00 - 17:00\",\"application_method\":\"\"}', 'Bachelor\'s degree in Computer Science or related field. Solid understanding of data structures, algorithms, and software design principles.', 'Phường Chánh Mỹ, Thành phố Thủ Dầu Một, Tỉnh Bình Dương', 2, '15,000,000 / tháng (Gross)', 1, '2025-04-13 16:45:32', '2025-04-30 00:00:00', 5, 'Java, SpringBoot, Mysql, PostgreSQL, Docker, CI/CD', '1', 1, 1),
(135, 1, 'PHP Developer', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-04-22 09:10:02', '2025-07-01 00:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(139, 1, 'PHP Developer', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-04-23 08:36:56', '2025-07-01 00:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(140, 1, 'Fullstack Developer', NULL, '{\"job_description\":{\"overview\":\"Lập trình API \",\"responsibilities\":[\"Lập trình API \",\"Lập trình API \",\"Lập trình API \"]},\"benefits\":{\"salary_range\":\"\",\"bonus\":\"13th month salary and annual performance-based bonus.\",\"additional_benefits\":[\"Không có\"]},\"work_hours\":\"9:00 - 18:00\",\"application_method\":\"\"}', 'Tốt nghiệp Đại học', 'Phường Chánh Mỹ, Thành phố Thủ Dầu Một, Tỉnh Bình Dương', 2, '10,000,000 / tháng', 1, '2025-04-23 08:38:52', '2025-04-29 17:00:00', 2, '.NET, C#, Golang, CI/CD, AWS', '2', 1, 3),
(141, 1, 'PHP Developer', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-04-27 14:47:49', '2025-07-01 00:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(142, 2, 'PHP Developer', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-04-28 01:30:54', '2025-07-01 00:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(143, 1, 'aaaa', NULL, '{\"job_description\":{\"overview\":\"aaa\",\"responsibilities\":[\"aaaa\"]},\"benefits\":{\"salary_range\":\"\",\"bonus\":\"13th month salary and annual performance-based bonus.\",\"additional_benefits\":[\"aaa\"]},\"work_hours\":\"8:00 - 17:00\",\"application_method\":\"\"}', 'aaa', 'Thành phố Hồ Chí Minh', 2, '15,000,000 / tháng (Gross)', 1, '2025-04-28 01:44:50', '2025-04-29 17:00:00', 1, 'Java, Spring, SQL, Postgresql, AWS, Docker, CI/CD', '1', 2, 1),
(195, 1, 'Java 3', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-07 19:38:32', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(216, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 22:10:25', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(217, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 22:16:58', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(218, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 22:47:31', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(219, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 22:59:16', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(220, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:10:55', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(221, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:11:12', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(222, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:19:09', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(223, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:19:15', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(224, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:20:10', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(225, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:21:08', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(226, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:21:50', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1);
INSERT INTO `job` (`id`, `employer_id`, `title`, `description`, `description_json`, `required`, `address`, `location_id`, `salary`, `status`, `posted_at`, `posted_expired`, `experience_id`, `required_skills`, `member`, `work_type_id`, `category_id`) VALUES
(227, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:38:41', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(228, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:40:14', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(229, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:51:56', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(230, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-09 23:52:43', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(231, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-10 00:02:15', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(232, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-10 00:02:25', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(233, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-12 00:21:47', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(234, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-12 12:31:00', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1),
(235, 1, 'Java 5', NULL, '{\"job_description\":{\"overview\":\"Join our team as a PHP Developer in Hà Nội. We are seeking a skilled PHP Developer to join our dynamic engineering team in Hanoi.\",\"responsibilities\":[\"Develop and maintain applications using PHP.\",\"Design and implement scalable RESTful APIs.\",\"Optimize server-side performance and ensure security.\",\"Collaborate with frontend developers for seamless integration.\"]},\"benefits\":{\"salary_range\":\"15,000,000 VND/month (net)\",\"bonus\":\"Performance-based bonus, up to 2 months\' salary.\",\"additional_benefits\":[\"Health insurance and social benefits as per Vietnamese law.\",\"Annual company trip and team-building activities.\",\"Training and certification support for Node.js technologies.\"]},\"work_hours\":\"Monday to Friday, 9:00 AM - 6:00 PM\",\"application_method\":\"Submit your CV and cover letter via email to hr@company.com with the subject \'PHP Developer Application - Hà Nội\'.\"}', 'Minimum 3 years of experience with PHP in production environments. Proficiency in MongoDB and RESTful API development.', 'Hà Nội', 1, '15000000', 1, '2025-06-12 12:40:57', '2025-07-01 07:00:00', 5, 'Node.js, Express, MongoDB, RESTful API', '1', 1, 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `location`
--

CREATE TABLE `location` (
  `id` int(11) NOT NULL,
  `name` text NOT NULL,
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `location`
--

INSERT INTO `location` (`id`, `name`, `status`) VALUES
(1, 'Hà Nội', 1),
(2, 'Hồ Chí Minh', 1),
(3, 'Đà Nẵng', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `matches`
--

CREATE TABLE `matches` (
  `id` int(11) NOT NULL,
  `cv_id` int(11) NOT NULL,
  `job_id` int(11) NOT NULL,
  `matched_skill` text NOT NULL,
  `time_matches` datetime NOT NULL,
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `matches`
--

INSERT INTO `matches` (`id`, `cv_id`, `job_id`, `matched_skill`, `time_matches`, `status`) VALUES
(105, 2, 85, 'javascript, html, css', '2025-05-08 20:55:08', 1),
(106, 2, 102, 'postgresql, java, spring, docker', '2025-05-08 20:55:08', 1),
(107, 2, 83, 'javascript, html, css', '2025-05-08 20:55:08', 1),
(108, 2, 94, 'mysql, php, laravel', '2025-05-08 20:55:08', 1),
(109, 2, 92, 'java, spring boot', '2025-05-08 20:55:08', 1),
(110, 2, 97, 'javascript, html, css', '2025-05-08 20:55:08', 1),
(111, 2, 96, 'postgresql, python', '2025-05-08 20:55:08', 1),
(112, 2, 95, '.net, sql server', '2025-05-08 20:55:08', 1),
(113, 2, 88, 'kafka', '2025-05-08 20:55:08', 1),
(114, 2, 82, 'postgresql', '2025-05-08 20:55:08', 1),
(115, 2, 89, 'python', '2025-05-08 20:55:08', 1),
(130, 2, 112, 'postgresql, mysql, java, docker', '2025-05-08 20:55:08', 1),
(131, 2, 110, 'postgresql, mysql, java, docker', '2025-05-08 20:55:08', 1),
(168, 2, 143, 'postgresql, java, spring, docker', '2025-05-08 20:55:08', 1),
(178, 2, 140, '.net', '2025-05-08 20:55:08', 1),
(271, 3, 85, 'javascript, css, html', '2025-06-12 12:42:47', 1),
(272, 3, 83, 'javascript, css, html', '2025-06-12 12:42:47', 1),
(273, 3, 143, 'java, docker, spring, postgresql', '2025-06-12 12:42:47', 1),
(274, 3, 102, 'java, docker, spring, postgresql', '2025-06-12 12:42:47', 1),
(275, 3, 112, 'java, mysql, docker, postgresql', '2025-06-12 12:42:47', 1),
(276, 3, 110, 'java, mysql, docker, postgresql', '2025-06-12 12:42:47', 1),
(277, 3, 94, 'mysql, laravel, php', '2025-06-12 12:42:47', 1),
(278, 3, 92, 'java, spring boot', '2025-06-12 12:42:47', 1),
(279, 3, 95, '.net, sql server', '2025-06-12 12:42:47', 1),
(280, 3, 97, 'javascript, css, html', '2025-06-12 12:42:47', 1),
(281, 3, 96, 'python, postgresql', '2025-06-12 12:42:47', 1),
(282, 3, 88, 'kafka', '2025-06-12 12:42:47', 1),
(283, 3, 82, 'postgresql', '2025-06-12 12:42:47', 1),
(284, 3, 140, '.net', '2025-06-12 12:42:47', 1),
(285, 3, 89, 'python', '2025-06-12 12:42:47', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `membership`
--

CREATE TABLE `membership` (
  `id` int(11) NOT NULL,
  `name` text NOT NULL,
  `price` double NOT NULL,
  `type_for` int(11) NOT NULL,
  `description` text NOT NULL,
  `duration` text NOT NULL,
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `membership`
--

INSERT INTO `membership` (`id`, `name`, `price`, `type_for`, `description`, `duration`, `status`) VALUES
(1, 'Gói miễn phí cho người ứng tuyển/ hàng tháng', 0, 1, 'Gói dịch vụ dành cho người tìm việc, Cung cấp quyền truy cập cơ bản miễn phí mỗi tháng, Bao gồm: tìm kiếm và xem thông tin việc làm, Ứng tuyển trực tiếp vào tin tuyển dụng, Quản lý hồ sơ cá nhân.', 'MONTHLY', 1),
(2, 'Gói miễn phí cho người ứng tuyển/ hàng năm', 0, 1, 'Gói dịch vụ dành cho người tìm việc, Cung cấp quyền truy cập cơ bản miễn phí mỗi tháng, Bao gồm: tìm kiếm và xem thông tin việc làm, Ứng tuyển trực tiếp vào tin tuyển dụng, Quản lý hồ sơ cá nhân.', 'YEARLY', 1),
(3, 'Gói PRO cho người ứng tuyển/ hàng tháng', 250000, 1, 'Gói PRO dành cho người tìm việc chuyên nghiệp, Mở rộng cơ hội với gợi ý việc làm thông minh bằng AI, Tích hợp và tối ưu CV dễ dàng, Thao tác ứng tuyển nhanh chóng, Nhận thông báo ngay khi có việc làm mới.', 'MONTHLY', 1),
(4, 'Gói PRO cho người ứng tuyển/ hàng năm', 2500000, 1, 'Gói PRO dành cho người tìm việc chuyên nghiệp, Mở rộng cơ hội với gợi ý việc làm thông minh bằng AI, Tích hợp và tối ưu CV dễ dàng, Thao tác ứng tuyển nhanh chóng, Nhận thông báo ngay khi có việc làm mới.', 'YEARLY', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `notifications`
--

CREATE TABLE `notifications` (
  `id` bigint(20) NOT NULL,
  `user_id` int(20) NOT NULL,
  `title` varchar(255) NOT NULL,
  `content` text NOT NULL,
  `is_read` tinyint(1) DEFAULT 0,
  `created_at` datetime DEFAULT current_timestamp(),
  `type` varchar(50) NOT NULL,
  `job_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `notifications`
--

INSERT INTO `notifications` (`id`, `user_id`, `title`, `content`, `is_read`, `created_at`, `type`, `job_id`) VALUES
(234, 95, 'Tin nhắn mới từ tài khoản <span class=\'sender-name\'>acc1</span>', 'Tài khoản <span class=\'sender-name\'>acc1</span> đã gửi tin nhắn: \"cc\"', 1, '2025-06-17 20:21:08', 'CHAT_MESSAGE', NULL),
(235, 1, 'Tin nhắn mới từ tài khoản <span class=\'sender-name\'>Tú Nguyễn</span>', 'Tài khoản <span class=\'sender-name\'>Tú Nguyễn</span> đã gửi tin nhắn: \"hi\"', 1, '2025-06-17 20:30:24', 'CHAT_MESSAGE', NULL),
(236, 1, 'Tin nhắn mới từ tài khoản <span class=\'sender-name\'>Tú Nguyễn</span>', 'Tài khoản <span class=\'sender-name\'>Tú Nguyễn</span> đã gửi tin nhắn: \"huhu\"', 1, '2025-06-17 20:34:26', 'CHAT_MESSAGE', NULL),
(237, 1, 'Tin nhắn mới từ tài khoản <span class=\'sender-name\'>Tú Nguyễn</span>', 'Tài khoản <span class=\'sender-name\'>Tú Nguyễn</span> đã gửi tin nhắn: \"khỏe không\"', 1, '2025-06-17 20:40:34', 'CHAT_MESSAGE', NULL),
(238, 1, 'Tin nhắn mới từ tài khoản <span class=\'sender-name\'>Tú Nguyễn</span>', 'Tài khoản <span class=\'sender-name\'>Tú Nguyễn</span> đã gửi tin nhắn: \"ccc\"', 1, '2025-06-17 20:43:41', 'CHAT_MESSAGE', NULL);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `payment`
--

CREATE TABLE `payment` (
  `id` int(11) NOT NULL,
  `employer_membership_id` int(11) DEFAULT NULL,
  `payment_type` text NOT NULL,
  `amount` double NOT NULL,
  `payment_method` text NOT NULL,
  `status` tinyint(1) NOT NULL,
  `transaction_id` int(11) NOT NULL,
  `description` text NOT NULL,
  `time` datetime NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `payment`
--

INSERT INTO `payment` (`id`, `employer_membership_id`, `payment_type`, `amount`, `payment_method`, `status`, `transaction_id`, `description`, `time`) VALUES
(1, 1, 'BANKING', 250000, 'VNPAY', 1, 240175, 'Thanh toán thành công số tiền 250000.0', '2025-04-10 10:05:46'),
(2, 4, 'BANKING', 2500000, 'VNPAY', 1, 361377, 'Thanh toán thành công số tiền 2500000.0', '2025-06-01 01:26:09');

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `question`
--

CREATE TABLE `question` (
  `id` int(11) NOT NULL,
  `testID` int(11) NOT NULL,
  `question_type` int(11) NOT NULL,
  `content` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `question`
--

INSERT INTO `question` (`id`, `testID`, `question_type`, `content`) VALUES
(1, 1, 1, '1 + 1 = ?'),
(3, 1, 2, 'Ai đẹp trai nhất'),
(4, 2, 1, 'Ai là tác giả bài thơ Việt Bắc'),
(5, 2, 1, 'Quang Trung là Nguyễn Huệ đúng hay sai'),
(6, 2, 2, 'Tả con mèo'),
(7, 3, 1, 'Hello, ... ?');

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `review`
--

CREATE TABLE `review` (
  `id` int(11) NOT NULL,
  `seeker_id` int(11) NOT NULL,
  `employer_id` int(11) NOT NULL,
  `rating` int(11) NOT NULL,
  `satisfied` tinyint(1) NOT NULL,
  `good_message` text NOT NULL,
  `reason` text NOT NULL,
  `improve` text NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `review`
--

INSERT INTO `review` (`id`, `seeker_id`, `employer_id`, `rating`, `satisfied`, `good_message`, `reason`, `improve`, `created_at`, `status`) VALUES
(2, 65, 1, 4, 1, 'cc', 'cc', 'ccc', '2025-04-06 09:47:05', 1),
(3, 87, 1, 5, 1, 'Môi trường và đồng nghiệp thân thiệt', 'Làm việc ở đây vui lắm', 'Không có gì để cải thiện !!!!', '2025-04-06 09:55:28', 1),
(4, 65, 1, 5, 0, 'Môi trường và đồng nghiệp thân thiệt', 'Môi trường và đồng nghiệp thân thiệt', 'Môi trường và đồng nghiệp thân thiệt', '2025-04-06 09:57:13', 1),
(5, 65, 1, 1, 0, 'ccccccccccccccccc', 'cccccccccccccccccccccccccccc', 'cccccccccccccccccccccc', '2025-04-06 09:57:39', 0),
(6, 65, 1, 5, 0, 'aaaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaaaaaaaa', '2025-04-06 09:58:12', 0),
(7, 65, 1, 4, 1, 'ccccccccccccccccccccc', 'cccccccccccccccccccccccccccccccc', 'cccccccccccccccccc', '2025-04-06 09:59:44', 0),
(8, 65, 1, 5, 1, 'Học hỏi được nhiều điều', 'Làm việc ở đây rất vui', 'Không có gì cả. Mọi thứ đều ổn', '2025-04-18 06:06:35', 0),
(9, 65, 1, 5, 1, 'Công ty rất tốt, đáng để học tập.', 'Công ty rất tốt, đáng để học tập.', 'Công ty rất tốt, đáng để học tập.', '2025-04-27 20:05:49', 0);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `seeker`
--

CREATE TABLE `seeker` (
  `id` int(11) NOT NULL,
  `full_name` text DEFAULT NULL,
  `phone` text DEFAULT NULL,
  `address` text DEFAULT NULL,
  `dob` date DEFAULT NULL,
  `gender` varchar(250) DEFAULT NULL,
  `status` tinyint(1) DEFAULT NULL,
  `update_at` datetime DEFAULT NULL,
  `avatar` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `seeker`
--

INSERT INTO `seeker` (`id`, `full_name`, `phone`, `address`, `dob`, `gender`, `status`, `update_at`, `avatar`) VALUES
(65, 'Hoàng Tú Nguyễn', '+84916700827', 'Phường Hùng Vương, Thị xã Phú Thọ, Tỉnh Phú Thọ', '2025-02-14', 'female', 1, NULL, '5e48b28ddd444d73837a2c1b25e2b7a8.jpg'),
(87, 'John Doe 123', '123456789', '123 Street', '2003-05-12', 'male', 1, NULL, 'c92231668dac49358afe7955a317e7b7.jpg'),
(95, 'Hoàng Tú Nguyễn', '+84916700827', 'Phạm Thế Hiển', '2031-05-20', NULL, 1, NULL, 'f28e7efe72c94264a62ecf96c9219de6.jpg');

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `skill`
--

CREATE TABLE `skill` (
  `id` int(11) NOT NULL,
  `name` text NOT NULL,
  `skill_parent_id` int(11) DEFAULT NULL,
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `test`
--

CREATE TABLE `test` (
  `id` int(11) NOT NULL,
  `title` text NOT NULL,
  `description` text NOT NULL,
  `userID` int(11) NOT NULL,
  `code` varchar(250) NOT NULL,
  `time` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `test`
--

INSERT INTO `test` (`id`, `title`, `description`, `userID`, `code`, `time`) VALUES
(1, 'Toán', 'aaa', 1, 'toan', 0),
(2, 'Ngữ Văn', 'aa', 1, 'nguvan', 0),
(3, 'Kiểm tra Java', 'Kiểm tra Java', 1, 'java', 10);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `testhistory`
--

CREATE TABLE `testhistory` (
  `id` int(11) NOT NULL,
  `testID` int(11) NOT NULL,
  `userID` int(11) NOT NULL,
  `time_submit` datetime DEFAULT NULL ON UPDATE current_timestamp(),
  `score` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `user`
--

CREATE TABLE `user` (
  `id` int(11) NOT NULL,
  `username` varchar(250) NOT NULL,
  `password` varchar(250) NOT NULL,
  `user_type` int(11) NOT NULL,
  `email` text NOT NULL,
  `created` date NOT NULL DEFAULT current_timestamp(),
  `security_code` text NOT NULL,
  `status` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `user`
--

INSERT INTO `user` (`id`, `username`, `password`, `user_type`, `email`, `created`, `security_code`, `status`) VALUES
(1, 'acc1', '$2a$10$dYXwldJJvKQsP8eeym8UI.Xv006GA.txAKYf9TtGVrJSXn7p5WbHO', 2, 'acc1@gmail.com', '2021-01-18', '', 1),
(2, 'fptsoftware', '$2a$10$dYXwldJJvKQsP8eeym8UI.Xv006GA.txAKYf9TtGVrJSXn7p5WbHO', 2, 'contact@fptsoftware.com', '2025-04-28', '123456', 1),
(3, 'tikicorporation', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'support@tiki.vn', '2025-05-02', '123456', 1),
(4, 'axonactive', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'info@axonactive.com', '2025-05-02', '123456', 1),
(5, 'kmstechnology', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'contact@kms-technology.com', '2025-05-02', '123456', 1),
(6, 'haravan', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'support@haravan.com', '2025-05-02', '123456', 1),
(7, 'tikinow', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'support@tiki.vn', '2025-05-02', '123456', 1),
(8, 'viettelsolutions', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'contact@viettelsolutions.vn', '2025-05-02', '123456', 1),
(9, 'tmasolutions', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'info@tmasolutions.com', '2025-05-02', '234567', 1),
(10, 'rikkeisoft', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'contact@rikkeisoft.com', '2025-05-02', '345678', 1),
(11, 'logigearvn', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'support@logigear.com', '2025-05-02', '456789', 1),
(12, 'tekexpertsvn', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'info@tekexperts.com', '2025-05-02', '567890', 1),
(13, 'suninc', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'contact@sun-asterisk.com', '2025-05-02', '678901', 1),
(14, 'nashtechvn', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'info@nashtechglobal.com', '2025-05-02', '789012', 1),
(15, 'fpttelecom', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'support@fpttelecom.vn', '2025-05-02', '890123', 1),
(16, 'cmcglobal', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'contact@cmcglobal.com', '2025-05-02', '901234', 1),
(17, 'smartosc', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'info@smartosc.com', '2025-05-02', '012345', 1),
(18, '10cloudsvietnam', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'contact@10clouds.com', '2025-05-02', '123456', 1),
(19, 'siiasiavn', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'info@siiasia.com', '2025-05-02', '234567', 1),
(20, 'digiworld', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'support@digiworld.com', '2025-05-02', '345678', 1),
(21, 'vinai', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'contact@vinai.io', '2025-05-02', '456789', 1),
(22, 'cyberlogitec', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'info@cyberlogitec.com', '2025-05-02', '567890', 1),
(23, 'globalcybersoft', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'support@globalcybersoft.com', '2025-05-02', '678901', 1),
(24, 'halotech', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'contact@halotech.vn', '2025-05-02', '789012', 1),
(25, 'kiotviet', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'info@kiotviet.vn', '2025-05-02', '890123', 1),
(26, 'vccorp', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'support@vccorp.vn', '2025-05-02', '901234', 1),
(27, 'ggroup', '$2a$10$EG2UXiZKO.GV.j2mM.0u4CA6BKV8Z897TNkZ...', 2, 'contact@ggroup.vn', '2025-05-02', '012345', 1),
(65, 'hoangtu', '$2a$10$dYXwldJJvKQsP8eeym8UI.Xv006GA.txAKYf9TtGVrJSXn7p5WbHO', 1, 'acc2@gmail.com', '2025-01-21', '741234', 1),
(66, 'nhà tuyển dụng', '$2a$10$c/LXuS9URRbeD6clfDH2nOBKYdp6goWh3Inemjdl9AMFEVmSg.Hsi', 2, 'test@gmail.com', '2023-04-12', '123456', 0),
(67, 'seeker 8', '$2a$10$nwaA.rUYFCwfjaNCDYD85O7ia1wEk1J/GWh/KUmv9AxVC/u8dZ1C6', 1, 'gmail1123', '2023-04-12', '123456', 0),
(71, 'seeker 9', '$2a$10$sDdFnQ6WUTGYdmUv2dUd6uHp/jsofA0mHi7U/Md/woKn7o9J3ZeD6', 1, 'gmail1123', '2023-04-12', '123456', 0),
(87, 'employee', '$2a$10$/6qPESzfhr8pfWD19qUw4e23LNfEaecet1CpkqiXORsOPzhlROCUO', 1, 'atun123456789cu@gmail.com', '2025-01-27', '607070', 1),
(88, 'Nhà tuyển dụng YES4ALL', '$2a$10$/6qPESzfhr8pfWD19qUw4e23LNfEaecet1CpkqiXORsOPzhlROCUO', 2, 'acc@gmail.com', '2025-04-23', '730979', 1),
(90, 'superadmin', '$2a$10$dYXwldJJvKQsP8eeym8UI.Xv006GA.txAKYf9TtGVrJSXn7p5WbHO', 0, 'superadmin@gmail.com', '2025-05-16', '123456', 1),
(95, 'Tú Nguyễn', '$2a$10$zAR9tclvTFdAScyNxSs6kOUaMz75xVOZ7G6/4o2lLK7Uce7T0JU1.', 1, 'acc2003@gmail.com', '2025-05-31', '430098', 1);

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `worktype`
--

CREATE TABLE `worktype` (
  `id` int(11) NOT NULL,
  `name` text NOT NULL,
  `status` tinyint(1) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Đang đổ dữ liệu cho bảng `worktype`
--

INSERT INTO `worktype` (`id`, `name`, `status`) VALUES
(1, 'Toàn thời gian', 1),
(2, 'Bán thời gian', 1),
(3, 'Thực tập', 1),
(4, 'Làm tại nhà', 1),
(5, 'Thời vụ', 1);

--
-- Chỉ mục cho các bảng đã đổ
--

--
-- Chỉ mục cho bảng `answer`
--
ALTER TABLE `answer`
  ADD PRIMARY KEY (`id`),
  ADD KEY `questionID` (`questionID`);

--
-- Chỉ mục cho bảng `application`
--
ALTER TABLE `application`
  ADD PRIMARY KEY (`id`),
  ADD KEY `seeker_id` (`seeker_id`),
  ADD KEY `job_id` (`job_id`);

--
-- Chỉ mục cho bảng `category`
--
ALTER TABLE `category`
  ADD PRIMARY KEY (`id`);

--
-- Chỉ mục cho bảng `chat`
--
ALTER TABLE `chat`
  ADD PRIMARY KEY (`id`),
  ADD KEY `receiver_id` (`receiver_id`),
  ADD KEY `sender_id` (`sender_id`);

--
-- Chỉ mục cho bảng `cv`
--
ALTER TABLE `cv`
  ADD PRIMARY KEY (`id`),
  ADD KEY `seeker_id` (`seeker_id`);

--
-- Chỉ mục cho bảng `employer`
--
ALTER TABLE `employer`
  ADD PRIMARY KEY (`id`);

--
-- Chỉ mục cho bảng `employermembership`
--
ALTER TABLE `employermembership`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `user_id` (`user_id`);

--
-- Chỉ mục cho bảng `experience`
--
ALTER TABLE `experience`
  ADD PRIMARY KEY (`id`);

--
-- Chỉ mục cho bảng `favorite`
--
ALTER TABLE `favorite`
  ADD PRIMARY KEY (`id`),
  ADD KEY `job_id` (`job_id`),
  ADD KEY `seeker_id` (`seeker_id`);

--
-- Chỉ mục cho bảng `feedback`
--
ALTER TABLE `feedback`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Chỉ mục cho bảng `follow`
--
ALTER TABLE `follow`
  ADD PRIMARY KEY (`id`),
  ADD KEY `employer_id` (`employer_id`),
  ADD KEY `seeker_id` (`seeker_id`);

--
-- Chỉ mục cho bảng `image`
--
ALTER TABLE `image`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Chỉ mục cho bảng `interview`
--
ALTER TABLE `interview`
  ADD PRIMARY KEY (`id`),
  ADD KEY `application_id` (`application_id`);

--
-- Chỉ mục cho bảng `job`
--
ALTER TABLE `job`
  ADD PRIMARY KEY (`id`),
  ADD KEY `employer_id` (`employer_id`),
  ADD KEY `experience_id` (`experience_id`),
  ADD KEY `location_id` (`location_id`),
  ADD KEY `work_type_id` (`work_type_id`),
  ADD KEY `category_id` (`category_id`);

--
-- Chỉ mục cho bảng `location`
--
ALTER TABLE `location`
  ADD PRIMARY KEY (`id`);

--
-- Chỉ mục cho bảng `matches`
--
ALTER TABLE `matches`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `unique_cv_job` (`cv_id`,`job_id`),
  ADD KEY `cv_id` (`cv_id`),
  ADD KEY `job_id` (`job_id`);

--
-- Chỉ mục cho bảng `membership`
--
ALTER TABLE `membership`
  ADD PRIMARY KEY (`id`);

--
-- Chỉ mục cho bảng `notifications`
--
ALTER TABLE `notifications`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Chỉ mục cho bảng `payment`
--
ALTER TABLE `payment`
  ADD PRIMARY KEY (`id`);

--
-- Chỉ mục cho bảng `question`
--
ALTER TABLE `question`
  ADD PRIMARY KEY (`id`),
  ADD KEY `testID` (`testID`);

--
-- Chỉ mục cho bảng `review`
--
ALTER TABLE `review`
  ADD PRIMARY KEY (`id`),
  ADD KEY `seeker_id` (`seeker_id`),
  ADD KEY `employer_id` (`employer_id`);

--
-- Chỉ mục cho bảng `seeker`
--
ALTER TABLE `seeker`
  ADD PRIMARY KEY (`id`);

--
-- Chỉ mục cho bảng `skill`
--
ALTER TABLE `skill`
  ADD PRIMARY KEY (`id`);

--
-- Chỉ mục cho bảng `test`
--
ALTER TABLE `test`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `code` (`code`),
  ADD KEY `userID` (`userID`);

--
-- Chỉ mục cho bảng `testhistory`
--
ALTER TABLE `testhistory`
  ADD PRIMARY KEY (`id`),
  ADD KEY `testID` (`testID`),
  ADD KEY `userID` (`userID`);

--
-- Chỉ mục cho bảng `user`
--
ALTER TABLE `user`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- Chỉ mục cho bảng `worktype`
--
ALTER TABLE `worktype`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT cho các bảng đã đổ
--

--
-- AUTO_INCREMENT cho bảng `answer`
--
ALTER TABLE `answer`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=17;

--
-- AUTO_INCREMENT cho bảng `application`
--
ALTER TABLE `application`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=101;

--
-- AUTO_INCREMENT cho bảng `category`
--
ALTER TABLE `category`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=95;

--
-- AUTO_INCREMENT cho bảng `chat`
--
ALTER TABLE `chat`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=99;

--
-- AUTO_INCREMENT cho bảng `cv`
--
ALTER TABLE `cv`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT cho bảng `employer`
--
ALTER TABLE `employer`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=89;

--
-- AUTO_INCREMENT cho bảng `employermembership`
--
ALTER TABLE `employermembership`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT cho bảng `experience`
--
ALTER TABLE `experience`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT cho bảng `favorite`
--
ALTER TABLE `favorite`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=24;

--
-- AUTO_INCREMENT cho bảng `feedback`
--
ALTER TABLE `feedback`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT cho bảng `follow`
--
ALTER TABLE `follow`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=21;

--
-- AUTO_INCREMENT cho bảng `image`
--
ALTER TABLE `image`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT cho bảng `interview`
--
ALTER TABLE `interview`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=29;

--
-- AUTO_INCREMENT cho bảng `job`
--
ALTER TABLE `job`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=236;

--
-- AUTO_INCREMENT cho bảng `location`
--
ALTER TABLE `location`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT cho bảng `matches`
--
ALTER TABLE `matches`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=286;

--
-- AUTO_INCREMENT cho bảng `membership`
--
ALTER TABLE `membership`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT cho bảng `notifications`
--
ALTER TABLE `notifications`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=239;

--
-- AUTO_INCREMENT cho bảng `payment`
--
ALTER TABLE `payment`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT cho bảng `question`
--
ALTER TABLE `question`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT cho bảng `review`
--
ALTER TABLE `review`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;

--
-- AUTO_INCREMENT cho bảng `seeker`
--
ALTER TABLE `seeker`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=96;

--
-- AUTO_INCREMENT cho bảng `skill`
--
ALTER TABLE `skill`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT cho bảng `test`
--
ALTER TABLE `test`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT cho bảng `testhistory`
--
ALTER TABLE `testhistory`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT cho bảng `user`
--
ALTER TABLE `user`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=96;

--
-- AUTO_INCREMENT cho bảng `worktype`
--
ALTER TABLE `worktype`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- Các ràng buộc cho các bảng đã đổ
--

--
-- Các ràng buộc cho bảng `answer`
--
ALTER TABLE `answer`
  ADD CONSTRAINT `answer_ibfk_1` FOREIGN KEY (`questionID`) REFERENCES `question` (`id`);

--
-- Các ràng buộc cho bảng `application`
--
ALTER TABLE `application`
  ADD CONSTRAINT `application_ibfk_1` FOREIGN KEY (`seeker_id`) REFERENCES `seeker` (`id`),
  ADD CONSTRAINT `application_ibfk_2` FOREIGN KEY (`job_id`) REFERENCES `job` (`id`);

--
-- Các ràng buộc cho bảng `chat`
--
ALTER TABLE `chat`
  ADD CONSTRAINT `chat_ibfk_1` FOREIGN KEY (`receiver_id`) REFERENCES `user` (`id`),
  ADD CONSTRAINT `chat_ibfk_2` FOREIGN KEY (`sender_id`) REFERENCES `user` (`id`);

--
-- Các ràng buộc cho bảng `cv`
--
ALTER TABLE `cv`
  ADD CONSTRAINT `cv_ibfk_1` FOREIGN KEY (`seeker_id`) REFERENCES `seeker` (`id`);

--
-- Các ràng buộc cho bảng `employer`
--
ALTER TABLE `employer`
  ADD CONSTRAINT `employer_ibfk_1` FOREIGN KEY (`id`) REFERENCES `user` (`id`);

--
-- Các ràng buộc cho bảng `favorite`
--
ALTER TABLE `favorite`
  ADD CONSTRAINT `favorite_ibfk_1` FOREIGN KEY (`job_id`) REFERENCES `job` (`id`),
  ADD CONSTRAINT `favorite_ibfk_2` FOREIGN KEY (`seeker_id`) REFERENCES `seeker` (`id`);

--
-- Các ràng buộc cho bảng `feedback`
--
ALTER TABLE `feedback`
  ADD CONSTRAINT `feedback_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`);

--
-- Các ràng buộc cho bảng `follow`
--
ALTER TABLE `follow`
  ADD CONSTRAINT `follow_ibfk_1` FOREIGN KEY (`employer_id`) REFERENCES `employer` (`id`),
  ADD CONSTRAINT `follow_ibfk_2` FOREIGN KEY (`seeker_id`) REFERENCES `seeker` (`id`);

--
-- Các ràng buộc cho bảng `image`
--
ALTER TABLE `image`
  ADD CONSTRAINT `image_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`);

--
-- Các ràng buộc cho bảng `interview`
--
ALTER TABLE `interview`
  ADD CONSTRAINT `interview_ibfk_1` FOREIGN KEY (`application_id`) REFERENCES `application` (`id`);

--
-- Các ràng buộc cho bảng `job`
--
ALTER TABLE `job`
  ADD CONSTRAINT `job_ibfk_1` FOREIGN KEY (`employer_id`) REFERENCES `employer` (`id`),
  ADD CONSTRAINT `job_ibfk_2` FOREIGN KEY (`experience_id`) REFERENCES `experience` (`id`),
  ADD CONSTRAINT `job_ibfk_3` FOREIGN KEY (`location_id`) REFERENCES `location` (`id`),
  ADD CONSTRAINT `job_ibfk_4` FOREIGN KEY (`work_type_id`) REFERENCES `worktype` (`id`),
  ADD CONSTRAINT `job_ibfk_5` FOREIGN KEY (`category_id`) REFERENCES `category` (`id`);

--
-- Các ràng buộc cho bảng `matches`
--
ALTER TABLE `matches`
  ADD CONSTRAINT `matches_ibfk_1` FOREIGN KEY (`cv_id`) REFERENCES `cv` (`id`),
  ADD CONSTRAINT `matches_ibfk_2` FOREIGN KEY (`job_id`) REFERENCES `job` (`id`);

--
-- Các ràng buộc cho bảng `notifications`
--
ALTER TABLE `notifications`
  ADD CONSTRAINT `notifications_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`);

--
-- Các ràng buộc cho bảng `question`
--
ALTER TABLE `question`
  ADD CONSTRAINT `question_ibfk_1` FOREIGN KEY (`testID`) REFERENCES `test` (`id`);

--
-- Các ràng buộc cho bảng `review`
--
ALTER TABLE `review`
  ADD CONSTRAINT `review_ibfk_1` FOREIGN KEY (`seeker_id`) REFERENCES `seeker` (`id`),
  ADD CONSTRAINT `review_ibfk_2` FOREIGN KEY (`employer_id`) REFERENCES `employer` (`id`);

--
-- Các ràng buộc cho bảng `seeker`
--
ALTER TABLE `seeker`
  ADD CONSTRAINT `seeker_ibfk_1` FOREIGN KEY (`id`) REFERENCES `user` (`id`);

--
-- Các ràng buộc cho bảng `test`
--
ALTER TABLE `test`
  ADD CONSTRAINT `test_ibfk_1` FOREIGN KEY (`userID`) REFERENCES `user` (`id`);

--
-- Các ràng buộc cho bảng `testhistory`
--
ALTER TABLE `testhistory`
  ADD CONSTRAINT `testhistory_ibfk_1` FOREIGN KEY (`testID`) REFERENCES `test` (`id`),
  ADD CONSTRAINT `testhistory_ibfk_2` FOREIGN KEY (`userID`) REFERENCES `user` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
