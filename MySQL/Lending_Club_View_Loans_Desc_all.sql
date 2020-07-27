CREATE VIEW loans_desc_all AS
SELECT * FROM lendingclub.loans_2007_2011
UNION ALL
SELECT * FROM lendingclub.loans_2012_2013