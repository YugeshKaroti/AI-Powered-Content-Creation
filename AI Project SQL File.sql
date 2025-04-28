create database AI;

use AI;

create table articles(article_id int unsigned primary key,
title text,
content json,
topic varchar(255),
person varchar(255),
style text,
model_used varchar(50),
created_at timestamp default current_timestamp, 
embedding_vectors longblob);

create table revision_history(revision_id int unsigned primary key,
    article_id int unsigned,
    revised_by varchar(50),
    previous_content json,
    revised_content json,
    revised_at timestamp default current_timestamp,
    foreign key(article_id) references articles(article_id) on delete cascade on update cascade);
    
create table metadata(metadata_id int unsigned primary key,
article_id int unsigned,
tags json,
reading_time float,
publication_status varchar(30),
foreign key (article_id) references articles(article_id) on delete cascade on update cascade);

create table model_performance(Test_id int unsigned primary key,
article_id int unsigned,
model_name varchar(50),
tested_at timestamp,
grammatical_score float,
publication_status varchar(50),
foreign key (article_id) references articles (article_id) on delete cascade on update cascade);

create table personality_templates(template_id int unsigned primary key,
article_id int unsigned,
template_name varchar(60),
`description` text,
prompt_template varchar(255),
foreign key(article_id) references articles(article_id) on delete cascade on update cascade);

create table sentence_embeddings(sentence_id int unsigned primary key,
sentence mediumtext,
embedding_vector longblob);

select * from articles;

select * from revision_history;

select * from metadata;

select * from model_performance;

select * from personality_templates;

select * from sentence_embeddings;