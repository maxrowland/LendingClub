LOAD DATA INFILE 'E:/DATABASES/LENDING_CLUB/LOAN_DATA/PREPROCESSED/loans_2020Q1.csv' 
REPLACE INTO TABLE lendingclub.loans_2020Q1
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(
	id,
    @member_id,
    @loan_amnt,
    @funded_amnt,
    @funded_amnt_inv,
    @term,
    @int_rate,
	@installment,
    @grade,
    @sub_grade,
    @emp_title,
    @emp_length,
    @home_ownership,
    @annual_inc,
    @verification_status,
    @issue_d,
    @loan_status,
    @pymnt_plan,
    @url,
    @`desc`,
    @purpose,
    @title,
    @zip_code,
    @addr_state,
    @dti,
    @delinq_2yrs,
    @earliest_cr_line,
    @fico_range_low,
    @fico_range_high,
    @inq_last_6mths,
    @mths_since_last_delinq,
    @mths_since_last_record,
    @open_acc,
    @pub_rec,
    @revol_bal,
    @revol_util,
    @total_acc,
    @initial_list_status,
    @out_prncp,
    @out_prncp_inv,
    @total_pymnt,
    @total_pymnt_inv,
    @total_rec_prncp,
    @total_rec_int,
    @total_rec_late_fee,
    @recoveries,
    @collection_recovery_fee,
    @last_pymnt_d,
    @last_pymnt_amnt,
    @next_pymnt_d,
    @last_credit_pull_d,
    @last_fico_range_high,
    @last_fico_range_low,
    @collections_12_mths_ex_med,
    @mths_since_last_major_derog,
    @policy_code,
    @application_type,
    @annual_inc_joint,
    @dti_joint,
    @verification_status_joint,
    @acc_now_delinq,
    @tot_coll_amt,
    @tot_cur_bal,
    @open_acc_6m,
    @open_act_il,
    @open_il_12m,
    @open_il_24m,
    @mths_since_rcnt_il,
    @total_bal_il,
    @il_util,
    @open_rv_12m,
    @open_rv_24m,
    @max_bal_bc,
    @all_util,
    @total_rev_hi_lim,
    @inq_fi,
    @total_cu_tl,
    @inq_last_12m,
    @acc_open_past_24mths,
    @avg_cur_bal,
    @bc_open_to_buy,
    @bc_util,
    @chargeoff_within_12_mths,
    @delinq_amnt,
    @mo_sin_old_il_acct,
    @mo_sin_old_rev_tl_op,
    @mo_sin_rcnt_rev_tl_op,
    @mo_sin_rcnt_tl,
    @mort_acc,
    @mths_since_recent_bc,
    @mths_since_recent_bc_dlq,
    @mths_since_recent_inq,
    @mths_since_recent_revol_delinq,
    @num_accts_ever_120_pd,
    @num_actv_bc_tl,
    @num_actv_rev_tl,
    @num_bc_sats,
    @num_bc_tl,
    @num_il_tl,
    @num_op_rev_tl,
    @num_rev_accts,
    @num_rev_tl_bal_gt_0,
    @num_sats,
    @num_tl_120dpd_2m,
    @num_tl_30dpd,
    @num_tl_90g_dpd_24m,
    @num_tl_op_past_12m,
    @pct_tl_nvr_dlq,
    @percent_bc_gt_75,
    @pub_rec_bankruptcies,
    @tax_liens,
    @tot_hi_cred_lim,
    @total_bal_ex_mort,
    @total_bc_limit,
    @total_il_high_credit_limit,
    @revol_bal_joint,
    @sec_app_fico_range_low,
    @sec_app_fico_range_high,
    @sec_app_earliest_cr_line,
    @sec_app_inq_last_6mths,
    @sec_app_mort_acc,
    @sec_app_open_acc,
    @sec_app_revol_util,
    @sec_app_open_act_il,
    @sec_app_num_rev_accts,
    @sec_app_chargeoff_within_12_mths,
    @sec_app_collections_12_mths_ex_med,
    @sec_app_mths_since_last_major_derog,
    @hardship_flag,
    @hardship_type,
    @hardship_reason,
    @hardship_status,
    @deferral_term,
    @hardship_amount,
    @hardship_start_date,
    @hardship_end_date,
    @payment_plan_start_date,
    @hardship_length,
    @hardship_dpd,
    @hardship_loan_status,
    @orig_projected_additional_accrued_interest,
    @hardship_payoff_balance_amount,
    @hardship_last_payment_amount,
    @debt_settlement_flag,
    @debt_settlement_flag_date,
    @settlement_status,
    @settlement_date,
    @settlement_amount,
    @settlement_percentage,
    @settlement_term)
    SET member_id = IF(@member_id = '', NULL, @member_id), 
    loan_amnt = IF(@loan_amnt = '', NULL, @loan_amnt),
    funded_amnt = IF(@funded_amnt = '', NULL, @funded_amnt),
    funded_amnt_inv = IF(@funded_amnt_inv = '', NULL, @funded_amnt_inv),
    term = IF(@term = '', NULL, @term),
    int_rate = IF(@int_rate = '', NULL, @int_rate),
	installment = IF(@installment = '', NULL, @installment),
    grade = IF(@grade = '', NULL, @grade),
    sub_grade = IF(@sub_grade = '', NULL, @sub_grade),
    emp_title = IF(@emp_title = '', NULL, @emp_title),
    emp_length = IF(@emp_length = '', NULL, @emp_length),
    home_ownership = IF(@home_ownership = '', NULL, @home_ownership),
    annual_inc = IF(@annual_inc = '', NULL, @annual_inc),
    verification_status = IF(@verification_status = '', NULL, @verification_status),
    issue_d = IF(@issue_d = '', NULL, @issue_d),
    loan_status = IF(@loan_status = '', NULL, @loan_status),
    pymnt_plan = IF(@pymnt_plan = '', NULL, @pymnt_plan),
    url = IF(@url = '', NULL, @url),
    `desc` = IF(@`desc` = '', NULL, @`desc`),
    purpose = IF(@purpose = '', NULL, @purpose),
    title = IF(@title = '', NULL, @title),
    zip_code = IF(@zip_code = '', NULL, @zip_code),
    addr_state = IF(@addr_state = '', NULL, @addr_state),
    dti = IF(@dti = '', NULL, @dti),
    delinq_2yrs = IF(@delinq_2yrs = '', NULL, @delinq_2yrs),
    earliest_cr_line = IF(@earliest_cr_line = '', NULL, @earliest_cr_line),
    fico_range_low = IF(@fico_range_low = '', NULL, @fico_range_low),
    fico_range_high = IF(@fico_range_high = '', NULL, @fico_range_high),
    inq_last_6mths = IF(@inq_last_6mths = '', NULL, @inq_last_6mths),
    mths_since_last_delinq = IF(@mths_since_last_delinq = '', NULL, @mths_since_last_delinq),
    mths_since_last_record = IF(@mths_since_last_record = '', NULL, @mths_since_last_record),
    open_acc = IF(@open_acc = '', NULL, @open_acc),
    pub_rec = IF(@pub_rec = '', NULL, @pub_rec),
    revol_bal = IF(@revol_bal = '', NULL, @revol_bal),
    revol_util = IF(@revol_util = '', NULL, @revol_util),
    total_acc = IF(@total_acc = '', NULL, @total_acc),
    initial_list_status = IF(@initial_list_status = '', NULL, @initial_list_status),
    out_prncp = IF(@out_prncp = '', NULL, @out_prncp),
    out_prncp_inv = IF(@out_prncp_inv = '', NULL, @out_prncp_inv),
    total_pymnt = IF(@total_pymnt = '', NULL, @total_pymnt),
    total_pymnt_inv = IF(@total_pymnt_inv = '', NULL, @total_pymnt_inv),
    total_rec_prncp = IF(@total_rec_prncp = '', NULL, @total_rec_prncp),
    total_rec_int = IF(@total_rec_int = '', NULL, @total_rec_int),
    total_rec_late_fee = IF(@total_rec_late_fee = '', NULL, @total_rec_late_fee),
    recoveries = IF(@recoveries = '', NULL, @recoveries),
    collection_recovery_fee = IF(@collection_recovery_fee = '', NULL, @collection_recovery_fee),
    last_pymnt_d = IF(@last_pymnt_d = '', NULL, @last_pymnt_d),
    last_pymnt_amnt = IF(@last_pymnt_amnt = '', NULL, @last_pymnt_amnt),
    next_pymnt_d = IF(@next_pymnt_d = '', NULL, @next_pymnt_d),
    last_credit_pull_d = IF(@last_credit_pull_d = '', NULL, @last_credit_pull_d),
    last_fico_range_high = IF(@last_fico_range_high = '', NULL, @last_fico_range_high),
    last_fico_range_low = IF(@last_fico_range_low = '', NULL, @last_fico_range_low),
    collections_12_mths_ex_med = IF(@collections_12_mths_ex_med = '', NULL, @collections_12_mths_ex_med),
    mths_since_last_major_derog = IF(@mths_since_last_major_derog = '', NULL, @mths_since_last_major_derog),
    policy_code = IF(@policy_code = '', NULL, @policy_code),
    application_type = IF(@application_type = '', NULL, @application_type),
    annual_inc_joint = IF(@annual_inc_joint = '', NULL, @annual_inc_joint),
    dti_joint = IF(@dti_joint = '', NULL, @dti_joint),
    verification_status_joint = IF(@verification_status_joint = '', NULL, @verification_status_joint),
    acc_now_delinq = IF(@acc_now_delinq = '', NULL, @acc_now_delinq),
    tot_coll_amt = IF(@tot_coll_amt = '', NULL, @tot_coll_amt),
    tot_cur_bal = IF(@tot_cur_bal = '', NULL, @tot_cur_bal),
    open_acc_6m = IF(@open_acc_6m = '', NULL, @open_acc_6m),
    open_act_il = IF(@open_act_il = '', NULL, @open_act_il),
    open_il_12m = IF(@open_il_12m = '', NULL, @open_il_12m),
    open_il_24m = IF(@open_il_24m = '', NULL, @open_il_24m),
    mths_since_rcnt_il = IF(@mths_since_rcnt_il = '', NULL, @mths_since_rcnt_il),
    total_bal_il = IF(@total_bal_il = '', NULL, @total_bal_il),
    il_util = IF(@il_util = '', NULL, @il_util),
    open_rv_12m = IF(@open_rv_12m = '', NULL, @open_rv_12m),
    open_rv_24m = IF(@open_rv_24m = '', NULL, @open_rv_24m),
    max_bal_bc = IF(@max_bal_bc = '', NULL, @max_bal_bc),
    all_util = IF(@all_util = '', NULL, @all_util),
    total_rev_hi_lim = IF(@total_rev_hi_lim = '', NULL, @total_rev_hi_lim),
    inq_fi = IF(@inq_fi = '', NULL, @inq_fi),
    total_cu_tl = IF(@total_cu_tl = '', NULL, @total_cu_tl),
    inq_last_12m = IF(@inq_last_12m = '', NULL, @inq_last_12m),
    acc_open_past_24mths = IF(@acc_open_past_24mths = '', NULL, @acc_open_past_24mths),
    avg_cur_bal = IF(@avg_cur_bal = '', NULL, @avg_cur_bal),
    bc_open_to_buy = IF(@bc_open_to_buy = '', NULL, @bc_open_to_buy),
    bc_util = IF(@bc_util = '', NULL, @bc_util),
    chargeoff_within_12_mths = IF(@chargeoff_within_12_mths = '', NULL, @chargeoff_within_12_mths),
    delinq_amnt = IF(@delinq_amnt = '', NULL, @delinq_amnt),
    mo_sin_old_il_acct = IF(@mo_sin_old_il_acct = '', NULL, @mo_sin_old_il_acct),
    mo_sin_old_rev_tl_op = IF(@mo_sin_old_rev_tl_op = '', NULL, @mo_sin_old_rev_tl_op),
    mo_sin_rcnt_rev_tl_op = IF(@mo_sin_rcnt_rev_tl_op = '', NULL, @mo_sin_rcnt_rev_tl_op),
    mo_sin_rcnt_tl = IF(@mo_sin_rcnt_tl = '', NULL, @mo_sin_rcnt_tl),
    mort_acc = IF(@mort_acc = '', NULL, @mort_acc),
    mths_since_recent_bc = IF(@mths_since_recent_bc = '', NULL, @mths_since_recent_bc),
    mths_since_recent_bc_dlq = IF(@mths_since_recent_bc_dlq = '', NULL, @mths_since_recent_bc_dlq),
    mths_since_recent_inq = IF(@mths_since_recent_inq = '', NULL, @mths_since_recent_inq),
    mths_since_recent_revol_delinq = IF(@mths_since_recent_revol_delinq = '', NULL, @mths_since_recent_revol_delinq),
    num_accts_ever_120_pd = IF(@num_accts_ever_120_pd = '', NULL, @num_accts_ever_120_pd),
    num_actv_bc_tl = IF(@num_actv_bc_tl = '', NULL, @num_actv_bc_tl),
    num_actv_rev_tl = IF(@num_actv_rev_tl = '', NULL, @num_actv_rev_tl),
    num_bc_sats = IF(@num_bc_sats = '', NULL, @num_bc_sats),
    num_bc_tl = IF(@num_bc_tl = '', NULL, @num_bc_tl),
    num_il_tl = IF(@num_il_tl = '', NULL, @num_il_tl),
    num_op_rev_tl = IF(@num_op_rev_tl = '', NULL, @num_op_rev_tl),
    num_rev_accts = IF(@num_rev_accts = '', NULL, @num_rev_accts),
    num_rev_tl_bal_gt_0 = IF(@num_rev_tl_bal_gt_0 = '', NULL, @num_rev_tl_bal_gt_0),
    num_sats = IF(@num_sats = '', NULL, @num_sats),
    num_tl_120dpd_2m = IF(@num_tl_120dpd_2m = '', NULL, @num_tl_120dpd_2m),
    num_tl_30dpd = IF(@num_tl_30dpd = '', NULL, @num_tl_30dpd),
    num_tl_90g_dpd_24m = IF(@num_tl_90g_dpd_24m = '', NULL, @num_tl_90g_dpd_24m),
    num_tl_op_past_12m = IF(@num_tl_op_past_12m = '', NULL, @num_tl_op_past_12m),
    pct_tl_nvr_dlq = IF(@pct_tl_nvr_dlq = '', NULL, @pct_tl_nvr_dlq),
    percent_bc_gt_75 = IF(@percent_bc_gt_75 = '', NULL, @percent_bc_gt_75),
    pub_rec_bankruptcies = IF(@pub_rec_bankruptcies = '', NULL, @pub_rec_bankruptcies),
    tax_liens = IF(@tax_liens = '', NULL, @tax_liens),
    tot_hi_cred_lim = IF(@tot_hi_cred_lim = '', NULL, @tot_hi_cred_lim),
    total_bal_ex_mort = IF(@total_bal_ex_mort = '', NULL, @total_bal_ex_mort),
    total_bc_limit = IF(@total_bc_limit = '', NULL, @total_bc_limit),
    total_il_high_credit_limit = IF(@total_il_high_credit_limit = '', NULL, @total_il_high_credit_limit),
    revol_bal_joint = IF(@revol_bal_joint = '', NULL, @revol_bal_joint),
    sec_app_fico_range_low = IF(@sec_app_fico_range_low = '', NULL, @sec_app_fico_range_low),
    sec_app_fico_range_high = IF(@sec_app_fico_range_high = '', NULL, @sec_app_fico_range_high),
    sec_app_earliest_cr_line = IF(@sec_app_earliest_cr_line = '', NULL, @sec_app_earliest_cr_line),
    sec_app_inq_last_6mths = IF(@sec_app_inq_last_6mths = '', NULL, @sec_app_inq_last_6mths),
    sec_app_mort_acc = IF(@sec_app_mort_acc = '', NULL, @sec_app_mort_acc),
    sec_app_open_acc = IF(@sec_app_open_acc = '', NULL, @sec_app_open_acc),
    sec_app_revol_util = IF(@sec_app_revol_util = '', NULL, @sec_app_revol_util),
    sec_app_open_act_il = IF(@sec_app_open_act_il = '', NULL, @sec_app_open_act_il),
    sec_app_num_rev_accts = IF(@sec_app_num_rev_accts = '', NULL, @sec_app_num_rev_accts),
    sec_app_chargeoff_within_12_mths = IF(@sec_app_chargeoff_within_12_mths = '', NULL, @sec_app_chargeoff_within_12_mths),
    sec_app_collections_12_mths_ex_med = IF(@sec_app_collections_12_mths_ex_med = '', NULL, @sec_app_collections_12_mths_ex_med),
    sec_app_mths_since_last_major_derog = IF(@sec_app_mths_since_last_major_derog = '', NULL, @sec_app_mths_since_last_major_derog),
    hardship_flag = IF(@hardship_flag = '', NULL, @hardship_flag),
    hardship_type = IF(@hardship_type = '', NULL, @hardship_type),
    hardship_reason = IF(@hardship_reason = '', NULL, @hardship_reason),
    hardship_status = IF(@hardship_status = '', NULL, @hardship_status),
    deferral_term = IF(@deferral_term = '', NULL, @deferral_term),
    hardship_amount = IF(@hardship_amount = '', NULL, @hardship_amount),
    hardship_start_date = IF(@hardship_start_date = '', NULL, @hardship_start_date),
    hardship_end_date = IF(@hardship_end_date = '', NULL, @hardship_end_date),
    payment_plan_start_date = IF(@payment_plan_start_date = '', NULL, @payment_plan_start_date),
    hardship_length = IF(@hardship_length = '', NULL, @hardship_length),
    hardship_dpd = IF(@hardship_dpd = '', NULL, @hardship_dpd),
    hardship_loan_status = IF(@hardship_loan_status = '', NULL, @hardship_loan_status),
    orig_projected_additional_accrued_interest = IF(@orig_projected_additional_accrued_interest = '', NULL, @orig_projected_additional_accrued_interest),
    hardship_payoff_balance_amount = IF(@hardship_payoff_balance_amount = '', NULL, @hardship_payoff_balance_amount),
    hardship_last_payment_amount = IF(@hardship_last_payment_amount = '', NULL, @hardship_last_payment_amount),
    debt_settlement_flag = IF(@debt_settlement_flag = '', NULL, @debt_settlement_flag),
    debt_settlement_flag_date = IF(@debt_settlement_flag_date = '', NULL, @debt_settlement_flag_date),
    settlement_status = IF(@settlement_status = '', NULL, @settlement_status),
    settlement_date = IF(@settlement_date = '', NULL, @settlement_date),
    settlement_amount = IF(@settlement_amount = '', NULL, @settlement_amount),
    settlement_percentage = IF(@settlement_percentage = '', NULL, @settlement_percentage),
    settlement_term = IF(@settlement_term = '', NULL, @settlement_term)
    ;

