CREATE TABLE IF NOT EXISTS soil_profile(
id serial primary key,
state varchar, district varchar, taluk varchar, village varchar, soil_type varchar, authority varchar,
ph float, ec float, oc float,
av_p float, av_K float,
av_s float, av_zn float, av_b float, av_fe float, av_cu float, av_mn float
);

"COPY soil_profile(
state, district, taluk, village, soil_type, authority,
ph, ec, oc, av_p, av_k,	av_s, av_zn, av_b, av_fe, av_cu,av_mn
)
FROM '/Users/saurabh/github/soil-classifier/dataset/soil-profile-data.csv' DELIMITER ',' CSV HEADER;"

db_user = sa
db_password = sa123

cat /Users/saurabh/github/soil-classifier/dataset/soil-profile-data.csv | psql -U sa -d soildb -h 127.0.0.1 -c
"COPY soil_profile(
state, district, taluk, village, soil_type, authority, ph, ec, oc, av_p, av_k,  av_s, av_zn, av_b, av_fe, av_cu,av_mn)
FROM stdin DELIMITER ',' CSV HEADER;"
Password for user sa:
COPY 92832








