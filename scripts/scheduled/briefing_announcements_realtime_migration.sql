-- ============================================
-- Briefing Announcements Delivery Split
-- ============================================
-- Adds independent Discord and voice delivery tracking so proactive briefings
-- can be posted to Discord without suppressing the voice assistant flow.
--
-- Run this in the Supabase SQL Editor.
-- ============================================

ALTER TABLE briefing_announcements
    ADD COLUMN IF NOT EXISTS discord_status TEXT NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS discord_sent_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS voice_status TEXT NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS voice_read_at TIMESTAMPTZ;

-- Backfill legacy rows that used the shared delivered/delivered_at fields.
UPDATE briefing_announcements
SET
    discord_status = CASE
        WHEN status = 'delivered' THEN 'sent'
        ELSE COALESCE(discord_status, 'pending')
    END,
    discord_sent_at = CASE
        WHEN status = 'delivered' THEN COALESCE(discord_sent_at, delivered_at, created_at)
        ELSE discord_sent_at
    END,
    voice_status = CASE
        WHEN status = 'delivered' THEN 'read'
        ELSE COALESCE(voice_status, 'pending')
    END,
    voice_read_at = CASE
        WHEN status = 'delivered' THEN COALESCE(voice_read_at, delivered_at, created_at)
        ELSE voice_read_at
    END
WHERE status = 'delivered'
   OR discord_status IS NULL
   OR voice_status IS NULL;

CREATE INDEX IF NOT EXISTS idx_briefings_discord_pending
ON briefing_announcements(user_id, status, discord_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_briefings_voice_pending
ON briefing_announcements(user_id, status, voice_status, created_at DESC);

COMMENT ON COLUMN briefing_announcements.discord_status IS 'Discord delivery state: pending or sent.';
COMMENT ON COLUMN briefing_announcements.discord_sent_at IS 'When the briefing was posted to Discord.';
COMMENT ON COLUMN briefing_announcements.voice_status IS 'Voice delivery state: pending or read.';
COMMENT ON COLUMN briefing_announcements.voice_read_at IS 'When the briefing was spoken by the voice assistant.';

-- Enable Supabase Realtime for briefing pushes to the Discord bot.
DO $$
BEGIN
    BEGIN
        EXECUTE 'ALTER PUBLICATION supabase_realtime ADD TABLE briefing_announcements';
    EXCEPTION
        WHEN duplicate_object THEN NULL;
        WHEN undefined_object THEN NULL;
    END;
END $$;
