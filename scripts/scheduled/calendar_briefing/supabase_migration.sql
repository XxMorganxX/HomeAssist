-- ============================================
-- Calendar Event Cache Table
-- ============================================
-- Stores processed calendar event IDs to prevent duplicate briefings.
-- Used by the calendar_briefing scheduled job.
--
-- Run this in Supabase SQL Editor to create the table.
-- ============================================

CREATE TABLE IF NOT EXISTS calendar_event_cache (
    event_id TEXT PRIMARY KEY,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for pruning old entries
CREATE INDEX IF NOT EXISTS idx_calendar_event_cache_first_seen 
ON calendar_event_cache(first_seen);

-- Comment for documentation
COMMENT ON TABLE calendar_event_cache IS 'Tracks processed Google Calendar events to avoid duplicate briefing announcements';
COMMENT ON COLUMN calendar_event_cache.event_id IS 'Google Calendar event ID';
COMMENT ON COLUMN calendar_event_cache.first_seen IS 'When this event was first processed';

