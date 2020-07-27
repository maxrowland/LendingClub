DROP VIEW IF EXISTS `loans_all`;
CREATE VIEW loans_all AS
SELECT * FROM lendingclub.loans_2007_2011
UNION ALL
SELECT * FROM lendingclub.loans_2012_2013
UNION ALL
SELECT * FROM lendingclub.loans_2014
UNION ALL
SELECT * FROM lendingclub.loans_2015
UNION ALL
SELECT * FROM lendingclub.loans_2016q1
UNION ALL
SELECT * FROM lendingclub.loans_2016q2
UNION ALL
SELECT * FROM lendingclub.loans_2016q3
UNION ALL
SELECT * FROM lendingclub.loans_2016q4
UNION ALL
SELECT * FROM lendingclub.loans_2017q1
UNION ALL
SELECT * FROM lendingclub.loans_2017q2
UNION ALL
SELECT * FROM lendingclub.loans_2017q3
UNION ALL
SELECT * FROM lendingclub.loans_2017q4
UNION ALL
SELECT * FROM lendingclub.loans_2018q1
UNION ALL
SELECT * FROM lendingclub.loans_2018q2
UNION ALL
SELECT * FROM lendingclub.loans_2018q3
UNION ALL
SELECT * FROM lendingclub.loans_2018q4
UNION ALL
SELECT * FROM lendingclub.loans_2019q1
UNION ALL
SELECT * FROM lendingclub.loans_2019q2
UNION ALL
SELECT * FROM lendingclub.loans_2019q3
UNION ALL
SELECT * FROM lendingclub.loans_2019q4
UNION ALL
SELECT * FROM lendingclub.loans_2020q1