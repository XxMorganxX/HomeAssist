-- ============================================
-- Test Briefing Data for Supabase
-- ============================================
-- Run this in Supabase SQL Editor to insert test briefings
-- that should be immediately announced on wake word.
-- ============================================

-- Clear any existing test briefings first (optional)
-- DELETE FROM briefing_announcements WHERE id LIKE 'test_%';

-- Insert a simple test briefing (no active_from - always active)
INSERT INTO briefing_announcements (id, user_id, content, opener_text, priority, status)
VALUES (
    'test_briefing_1',
    'Morgan',
    '{"message": "This is a test briefing to verify the system works.", "llm_instructions": "Just say this is a test.", "meta": {"source": "test"}}',
    'Hey! Just a quick test to make sure briefings are working properly. Is there anything I can help you with?',
    'normal',
    'pending'
)
ON CONFLICT (id) DO UPDATE SET
    status = 'pending',
    opener_text = EXCLUDED.opener_text,
    delivered_at = NULL;

-- Insert a calendar-style briefing with active_from in the past (should be active now)
INSERT INTO briefing_announcements (id, user_id, content, opener_text, priority, status)
VALUES (
    'test_calendar_reminder_1',
    'Morgan',
    '{"message": "You have a team meeting coming up in 30 minutes.", "llm_instructions": "Remind the user about their meeting.", "active_from": "2020-01-01T00:00:00Z", "meta": {"source": "calendar_briefing", "event_title": "Team Meeting"}}',
    'Heads up! Your team meeting is coming up in about 30 minutes. Need anything before then?',
    'high',
    'pending'
)
ON CONFLICT (id) DO UPDATE SET
    status = 'pending',
    opener_text = EXCLUDED.opener_text,
    delivered_at = NULL;

-- Verify the data was inserted
SELECT 
    id,
    user_id,
    status,
    priority,
    LEFT(opener_text, 50) as opener_preview,
    content->>'active_from' as active_from,
    created_at
FROM briefing_announcements 
WHERE user_id = 'Morgan' AND status = 'pending'
ORDER BY created_at DESC;

