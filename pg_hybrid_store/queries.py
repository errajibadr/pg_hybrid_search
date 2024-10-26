LIST_VECTOR_STORES_QUERY = """
WITH vector_tables AS (
    SELECT
        t.table_schema,
        t.table_name,
        c.column_name,
        pg_size_pretty(pg_total_relation_size(t.table_name::regclass)) as table_size,
        get_table_row_count(t.table_schema, t.table_name) as vector_count,
        EXISTS (
            SELECT 1
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_am am ON am.oid = c.relam
            WHERE am.amname = 'diskann'
            AND i.indrelid = (t.table_schema || '.' || t.table_name)::regclass
        ) as has_diskann_index
    FROM information_schema.tables t
    JOIN information_schema.columns c
        ON c.table_name = t.table_name
        AND c.table_schema = t.table_schema
    WHERE c.udt_name = 'vector'
    AND t.table_schema = 'public'
),
vector_dimensions AS (
  SELECT
    a.attrelid::regclass::text AS table_name,
    a.attname AS column_name,
    CASE
      WHEN t.typname = 'vector' THEN (a.atttypmod)
    END AS vector_dimension
  FROM pg_attribute a
  JOIN pg_type t ON a.atttypid = t.oid
  WHERE t.typname = 'vector'
  AND a.attnum > 0  -- Skip system columns
)
select vt.*,vd.vector_dimension from vector_tables vt
join vector_dimensions vd on vt.table_name = vd.table_name
"""

LIST_DISKANN_INDEXES_QUERY = """
    SELECT
        i.indrelid::regclass AS table_name
        FROM
            pg_index i
        JOIN
            pg_class c ON c.oid = i.indexrelid
        JOIN
            pg_am am ON am.oid = c.relam
        WHERE
            am.amname = 'diskann';
"""
