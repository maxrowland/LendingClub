DROP VIEW IF EXISTS `loans_charge_off`;
CREATE VIEW loans_charge_off AS
SELECT
LOAN_ID,
`MONTH`,
CHARGE_OFF
FROM lendingclub.pmthist_all