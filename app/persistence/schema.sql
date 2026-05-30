-- 项目数据库结构（SQLite）。几何统一用 WKT(LINESTRING) 存储。
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS drawing (
    id          TEXT PRIMARY KEY,
    image_path  TEXT NOT NULL,
    width_px    INTEGER NOT NULL,
    height_px   INTEGER NOT NULL,
    crop_x      INTEGER, crop_y INTEGER, crop_w INTEGER, crop_h INTEGER
);

CREATE TABLE IF NOT EXISTS pipeline (
    id           TEXT PRIMARY KEY,
    drawing_id   TEXT NOT NULL REFERENCES drawing(id) ON DELETE CASCADE,
    name         TEXT,
    geometry_wkt TEXT,
    state        TEXT DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS length_label (
    id                 TEXT PRIMARY KEY,
    drawing_id         TEXT NOT NULL REFERENCES drawing(id) ON DELETE CASCADE,
    text               TEXT,
    value_m            REAL,
    bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER,
    confidence         REAL,
    matched_segment_id TEXT,
    state              TEXT DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS calibration (
    id            TEXT PRIMARY KEY,
    pipeline_id   TEXT REFERENCES pipeline(id) ON DELETE CASCADE,
    pixel_length  REAL,
    real_length_m REAL,
    scale_m_per_px REAL,
    source        TEXT,
    confidence    REAL
);

CREATE TABLE IF NOT EXISTS failure_point (
    id          TEXT PRIMARY KEY,
    drawing_id  TEXT NOT NULL REFERENCES drawing(id) ON DELETE CASCADE,
    code        TEXT,
    raw_x REAL, raw_y REAL, proj_x REAL, proj_y REAL,
    pipeline_id TEXT,
    offset_px   REAL,
    marker_type TEXT,
    confidence  REAL,
    state       TEXT DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS measure_result (
    id            TEXT PRIMARY KEY,
    drawing_id    TEXT NOT NULL REFERENCES drawing(id) ON DELETE CASCADE,
    from_code     TEXT, to_code TEXT, pipeline_name TEXT,
    straight_px   REAL, along_pipe_px REAL,
    straight_m    REAL, along_pipe_m REAL,
    basis         TEXT, confidence REAL, need_review INTEGER
);

CREATE INDEX IF NOT EXISTS idx_measure_result_drawing ON measure_result(drawing_id);
CREATE INDEX IF NOT EXISTS idx_failure_point_drawing ON failure_point(drawing_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_drawing ON pipeline(drawing_id);

CREATE TABLE IF NOT EXISTS review_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT,
    entity_type TEXT, entity_id TEXT, action TEXT, detail TEXT
);
