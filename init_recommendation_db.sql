DROP DATABASE IF EXISTS crowstream_recommendation_db;
CREATE DATABASE crowstream_recommendation_db;
CREATE USER crowstream_recommendation_ms WITH PASSWORD 'crowstream2021';
GRANT ALL PRIVILEGES ON DATABASE crowstream_recommendation_db to crowstream_recommendation_ms;