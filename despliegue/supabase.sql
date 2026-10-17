-- Base de datos de respuestas "¿Acertamos?" de Estavoz.
-- Pegar en Supabase: SQL Editor -> New query -> Run.

create table if not exists respuestas (
  id         bigint generated always as identity primary key,
  creado     timestamptz not null default now(),
  propuesto  text not null,              -- actor que dijo la app
  acierto    boolean not null,           -- ¿el usuario dice que acertó?
  real       text,                       -- si no acertó y el usuario sabe quién era
  huellas    jsonb not null,             -- huellas de voz de la grabación (nunca el audio)
  version    text
);

alter table respuestas enable row level security;

-- La web (clave pública "anon") solo puede AÑADIR respuestas, con límites de tamaño.
-- Leerlas solo se puede con la clave secreta (service_role), desde nuestro programa.
drop policy if exists "la web solo puede añadir" on respuestas;
create policy "la web solo puede añadir" on respuestas
  for insert to anon
  with check (
    length(propuesto) < 100
    and (real is null or length(real) < 100)
    and jsonb_typeof(huellas) = 'array'
    and jsonb_array_length(huellas) between 1 and 4
    and octet_length(huellas::text) < 20000
  );
