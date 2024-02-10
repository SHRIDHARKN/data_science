# MySQL
```SQL
-- create a database
create schema evilcorp;

-- create a table in the schema
create table evilcorp.users (
    id int auto_increment primary key,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    auth_key VARCHAR(100) NOT NULL
);

-- insert records into the table
-- NOTE only the name, mail, pwd is required to be entered
insert into  evilcorp.users (username, email, auth_key) VALUES
('white rose', 'whiterose@evilcorp.com', '$hgyd^&**'),
('philip price', 'philipprice@evilcorp.com', '$%^$%%FFSDGF'),
('terry colby', 'terrycolby@evilcorp.com', '()(()(@#$##*({}]');
```
