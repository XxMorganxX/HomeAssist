-- ============================================
-- Todos Table
-- ============================================
-- Persistent task storage for voice, Discord, and calendar-linked todos.
-- Run this in the Supabase SQL Editor.
-- ============================================

CREATE TABLE IF NOT EXISTS todos (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    details TEXT,
    due_at TIMESTAMPTZ,
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    completed_at TIMESTAMPTZ,
    source_type TEXT NOT NULL,
    source_id TEXT,
    source_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_todos_user_completed_due
ON todos(user_id, completed, due_at);

CREATE INDEX IF NOT EXISTS idx_todos_created_at
ON todos(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_todos_source
ON todos(user_id, source_type, source_id);

CREATE INDEX IF NOT EXISTS idx_todos_source_metadata
ON todos USING gin (source_metadata);

COMMENT ON TABLE todos IS 'Persistent action items created from voice, Discord, manual commands, and calendar-linked sources.';
COMMENT ON COLUMN todos.source_type IS 'Origin of the todo such as voice, discord, calendar, or manual.';
COMMENT ON COLUMN todos.source_id IS 'External or generated identifier for the todo source.';
COMMENT ON COLUMN todos.source_metadata IS 'Source-specific context such as calendar event details or Discord interaction metadata.';
