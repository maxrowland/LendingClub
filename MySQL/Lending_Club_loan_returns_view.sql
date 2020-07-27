DROP VIEW IF EXISTS `loan_returns`;
CREATE VIEW loan_returns AS
SELECT
LOAN_ID,
`MONTH`,
NET_RETURN
FROM lendingclub.pmthist_all