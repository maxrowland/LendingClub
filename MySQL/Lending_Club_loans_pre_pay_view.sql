DROP VIEW IF EXISTS `loans_pre_pay`;
CREATE VIEW loans_pre_pay AS
SELECT
LOAN_ID,
`MONTH`,
PRE_PAY
FROM lendingclub.pmthist_all