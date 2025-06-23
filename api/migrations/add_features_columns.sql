-- Migration script to add features columns to CV and Job tables
-- Run this script to add the new columns for storing extracted features

-- Add features columns to CV table
ALTER TABLE `cv` 
ADD COLUMN `primary_skills` JSON DEFAULT NULL COMMENT 'Primary skills extracted from CV',
ADD COLUMN `secondary_skills` JSON DEFAULT NULL COMMENT 'Secondary skills extracted from CV',
ADD COLUMN `adverbs` JSON DEFAULT NULL COMMENT 'Adverbs extracted from CV',
ADD COLUMN `adjectives` JSON DEFAULT NULL COMMENT 'Adjectives extracted from CV';

-- Add features columns to Job table
ALTER TABLE `job` 
ADD COLUMN `primary_skills` JSON DEFAULT NULL COMMENT 'Primary skills extracted from Job Description',
ADD COLUMN `secondary_skills` JSON DEFAULT NULL COMMENT 'Secondary skills extracted from Job Description',
ADD COLUMN `adverbs` JSON DEFAULT NULL COMMENT 'Adverbs extracted from Job Description',
ADD COLUMN `adjectives` JSON DEFAULT NULL COMMENT 'Adjectives extracted from Job Description';

-- Create indexes for better performance (optional)
-- Note: MySQL doesn't support direct indexing on JSON columns, but you can create functional indexes if needed

-- Verify the changes
DESCRIBE `cv`;
DESCRIBE `job`;
