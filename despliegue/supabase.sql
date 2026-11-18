-- Base de datos de respuestas "¿Acertamos?" de Estavoz.
-- Pegar en Supabase: SQL Editor -> New query -> Run.
-- Script idempotente: crea la tabla en instalaciones nuevas y migra tablas existentes sin pérdida de datos.

create table if not exists respuestas (
  id         bigint generated always as identity primary key,
  creado     timestamptz not null default now(),
  propuesto  text not null,              -- actor que dijo la app
  acierto    boolean not null,           -- ¿el usuario dice que acertó?
  real       text,                       -- si no acertó y el usuario sabe quién era
  huellas    jsonb not null,             -- huellas de voz de la grabación (nunca el audio)
  version    text,
  estado     text,                       -- 'identificado' | 'dudoso' | 'desconocido'
  parecido   real,                       -- puntuación AS-Norm del candidato propuesto
  f0         real,                       -- frecuencia fundamental estimada de la consulta (Hz)
  tomas      smallint,                   -- número de tomas acumuladas en la consulta
  con_pista  boolean                     -- si había filtro contextual activo (sin guardar el título)
);

-- Migración idempotente para bases de datos ya creadas en versiones anteriores
alter table respuestas add column if not exists version text;
alter table respuestas add column if not exists estado text;
alter table respuestas add column if not exists parecido real;
alter table respuestas add column if not exists f0 real;
alter table respuestas add column if not exists tomas smallint;
alter table respuestas add column if not exists con_pista boolean;

create index if not exists idx_respuestas_version_id on respuestas (version, id);
create index if not exists idx_respuestas_creado on respuestas (creado desc);

alter table respuestas enable row level security;

grant usage on schema public to anon;
grant insert on table respuestas to anon;

-- La web (clave pública "anon") solo puede AÑADIR respuestas, con límites de tamaño y rango.
-- Leerlas solo se puede con la clave secreta (service_role), desde nuestro programa o el SQL Editor.
drop policy if exists "la web solo puede añadir" on respuestas;
create policy "la web solo puede añadir" on respuestas
  for insert to anon
  with check (
    char_length(trim(propuesto)) between 1 and 100
    and (real is null or char_length(trim(real)) between 1 and 100)
    and (not acierto or real is null)
    and (version is null or char_length(trim(version)) between 1 and 24)
    and jsonb_typeof(huellas) = 'array'
    and jsonb_array_length(huellas) between 1 and 4
    and octet_length(huellas::text) < 20000
    and (estado is null or estado in ('identificado', 'dudoso', 'desconocido'))
    and (parecido is null or (parecido >= -1 and parecido <= 2))
    and (f0 is null or (f0 >= 50 and f0 <= 400))
    and (tomas is null or tomas between 1 and 10)
  );

-- Vista agregada de seguimiento por versión y estado (solo lectura para SQL Editor / service_role)
create or replace view respuestas_resumen with (security_invoker = true) as
  select
    coalesce(version, 'sin_version') as version,
    coalesce(estado, 'sin_estado') as estado,
    count(*) as total,
    count(*) filter (where acierto) as aciertos,
    count(*) filter (where not acierto and real is not null) as con_correccion,
    round(avg(parecido)::numeric, 4) as parecido_medio,
    max(creado) as ultima_respuesta
  from respuestas
  group by 1, 2
  order by 1 desc, 3 desc;

revoke all on respuestas_resumen from public, anon, authenticated;
grant select on respuestas_resumen to service_role;
