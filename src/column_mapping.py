"""
column_mapping.py
-----------------
Maps the original Taiwanese bankruptcy dataset column names (encoded/corrupted)
to clean, standardized English feature names used throughout the system.

Also defines which features are CRITICAL (required) vs OPTIONAL for prediction.
"""

# Maps original column name → clean English name
COLUMN_MAPPING = {
    "Flag":                                                                  "bankruptcy_flag",

    # --- Profitability Ratios ---
    "ROA(C)µ|«e®§«e§éÂÂ«e":                                                "roa_before_tax_c",
    "ROA(A)µ|«á®§«e%":                                                      "roa_after_tax_a",
    "ROA(B)µ|«á®§«e§éÂÂ«e":                                                 "roa_after_tax_b",
    "Àç·~¤ò§Q²v":                                                           "operating_gross_margin",
    "¤w¹ê²{¾P³f¤ò§Q²v":                                                     "realized_sales_gross_margin",
    "Àç·~§Q¯q²v":                                                           "operating_profit_rate",
    "µ|«e²b§Q²v":                                                           "pre_tax_net_profit_rate",
    "µ|«á²b§Q²v":                                                           "after_tax_net_profit_rate",
    "·~¥~¦¬¤ä /Àç¦¬":                                                       "non_operating_income_to_revenue",
    "±`Äò§Q¯q²v (µ|«á)":                                                    "continuous_profit_rate_after_tax",
    "Àç·~¶O¥Î²v":                                                           "operating_expense_rate",
    "¬ã¨sµo®i¶O¥Î²v":                                                       "research_development_expense_rate",

    # --- Cash Flow Ratios ---
    "²{ª÷¬y¶q¤ñ²v":                                                         "cash_flow_rate",
    "¦³®§­t¶Å§Q²v":                                                         "interest_bearing_debt_ratio",

    # --- Tax & Per-Share Metrics ---
    "µ|²v (A)":                                                             "tax_rate_a",
    "¨CªÑ²b­È (B)":                                                         "net_value_per_share_b",
    "¨CªÑ²b­È (A)":                                                         "net_value_per_share_a",
    "¨CªÑ²b­È (C)":                                                         "net_value_per_share_c",
    "ªñ¥|©u±`Äò©ÊEPS":                                                      "eps_last_four_quarters",
    "¨CªÑ²{ª÷¬y¶q":                                                         "cash_flow_per_share",
    "¨CªÑÀç·~ÃB(¤¸)":                                                       "revenue_per_share",
    "¨CªÑÀç·~§Q¯q(¤¸)":                                                     "operating_profit_per_share",
    "¨CªÑµ|«e²b§Q(¤¸)":                                                     "pre_tax_net_profit_per_share",

    # --- Growth Rates ---
    "¤w¹ê²{¾P³f¤ò§Q¦¨ªø²v":                                                "realized_sales_gross_profit_growth",
    "Àç·~§Q¯q¦¨ªø²v":                                                       "operating_profit_growth",
    "µ|«á²b§Q¦¨ªø²v":                                                       "after_tax_net_profit_growth",
    "¸g±`²b§Q¦¨ªø²v":                                                       "regular_net_profit_growth",
    "±`Äò²b§Q¦¨ªø²v":                                                       "continuous_net_profit_growth",
    "Á`¸ê²£¦¨ªø²v":                                                         "total_assets_growth",
    "²b­È¦¨ªø²v":                                                           "net_value_growth",
    "Á`¸ê²£³ø¹S¦¨ªø²v":                                                     "total_asset_return_growth",
    "²{ª÷¦A§ë¸ê%":                                                          "cash_reinvestment_pct",

    # --- Liquidity Ratios ---
    "¬y°Ê¤ñ²v":                                                             "current_ratio",
    "³t°Ê¤ñ²v":                                                             "quick_ratio",
    "§Q®§¤ä¥X²v":                                                           "interest_expense_ratio",

    # --- Leverage / Debt Ratios ---
    "Á`­t¶Å/Á`²b­È":                                                        "total_debt_to_net_value",
    "­t¶Å¤ñ²v¢H":                                                            "debt_ratio",
    "²b­È /¸ê²£":                                                            "net_value_to_assets",
    "ªø´Á¸êª÷¾A¦X²v (A)":                                                   "long_term_fund_suitability_a",
    "­É´Ú¨Ì¦s«×":                                                            "borrowing_dependency",
    "©Î¦³­t¶Å /²b­È":                                                       "contingent_liabilities_to_net_value",
    "Àç·~§Q¯q /¹ê¦¬¸ê¥»":                                                   "operating_profit_to_paid_in_capital",
    "µ|«e¯Â¯q /¹ê¦¬¸ê¥»":                                                   "pre_tax_net_profit_to_paid_in_capital",
    "¦s³f¤ÎÀ³¦¬±b´Ú /²b­È":                                                 "inventory_and_receivables_to_net_value",

    # --- Turnover / Efficiency Ratios ---
    "Á`¸ê²£¶gÂà¦¸¼Æ":                                                       "total_asset_turnover",
    "À³¦¬±b´Ú¶gÂà¦¸":                                                       "accounts_receivable_turnover",
    "¥­§¡¦¬±b¤Ñ¼Æ":                                                         "average_collection_days",
    "¦s³f¶gÂà²v (¦¸)":                                                      "inventory_turnover",
    "©T©w¸ê²£¶gÂà¦¸¼Æ":                                                     "fixed_assets_turnover",
    "²b­È¶gÂà²v (¦¸)":                                                      "net_value_turnover",

    # --- Per-Employee Metrics ---
    "¨C¤HÀç¦¬":                                                             "revenue_per_employee",
    "¨C¤HÀç·~§Q¯q":                                                         "operating_profit_per_employee",
    "¨C¤H°t³Æ²v":                                                           "allocation_rate_per_employee",

    # --- Features Already in English (indices 54-95) ---
    "working capital to total assets":                                       "working_capital_to_total_assets",
    "Quick asset/Total asset":                                               "quick_asset_to_total_asset",
    "current assets/total assets":                                           "current_assets_to_total_assets",
    "cash / total assets":                                                   "cash_to_total_assets",
    "Quick asset/current liabilities":                                       "quick_asset_to_current_liabilities",
    "cash / current liability":                                              "cash_to_current_liability",
    "current liability to assets":                                           "current_liability_to_assets",
    "operating funds to liability":                                          "operating_funds_to_liability",
    "Inventory/working capital":                                             "inventory_to_working_capital",
    "Inventory/current liability":                                           "inventory_to_current_liability",
    "current liability / liability":                                         "current_liability_to_total_liability",
    "working capital/equity":                                                "working_capital_to_equity",
    "current liability/equity":                                              "current_liability_to_equity",
    "long-term liability to current assets":                                 "long_term_liability_to_current_assets",
    "Retained Earnings/Total assets":                                        "retained_earnings_to_total_assets",
    "total income / total expense":                                          "total_income_to_total_expense",
    "total expense /assets":                                                 "total_expense_to_assets",
    "¬y°Ê¸ê²£©PÂà²v":                                                       "current_assets_turnover_rate",
    "³t°Ê¸ê²£©PÂà²v":                                                       "quick_assets_turnover_rate",
    "working capitcal  ©PÂà²v":                                             "working_capital_turnover_rate",
    "²{ª÷©PÂà²v":                                                           "cash_turnover_rate",
    "Cash flow to Sales":                                                    "cash_flow_to_sales",
    "fix assets to assets":                                                  "fixed_assets_to_total_assets",
    "current liability to liability":                                        "current_liability_to_liability",
    "current liability to equity":                                           "current_liability_to_equity_b",
    "equity to long-term liability":                                         "equity_to_long_term_liability",
    "Cash flow to total assets":                                             "cash_flow_to_total_assets",
    "cash flow to liability":                                                "cash_flow_to_liability",
    "CFO to ASSETS":                                                         "cfo_to_assets",
    "cash flow to equity":                                                   "cash_flow_to_equity",
    "current liabilities to current assets":                                 "current_liabilities_to_current_assets",
    "one if total liabilities exceeds total assets  zero otherwise":         "liabilities_exceed_assets_flag",
    "net income to total assets":                                            "net_income_to_total_assets",
    "total assets to GNP price":                                             "total_assets_to_gnp_price",
    "No-credit interval":                                                    "no_credit_interval",
    "Gross profit to Sales":                                                 "gross_profit_to_sales",
    "Net income to stockholder's Equity":                                    "net_income_to_equity",
    "liability to equity":                                                   "liability_to_equity",
    "Degree of financial leverage (DFL)":                                    "degree_of_financial_leverage",
    "Interest coverage ratio( Interest expense to EBIT )":                  "interest_coverage_ratio",
    "one if net income was negative for the last two year  zero otherwise":  "net_income_negative_two_years_flag",
    "equity to liability":                                                   "equity_to_liability",
}

# Features grouped by category for the UI form
FEATURE_GROUPS = {
    "Profitability": [
        "roa_before_tax_c", "roa_after_tax_a", "roa_after_tax_b",
        "operating_gross_margin", "realized_sales_gross_margin",
        "operating_profit_rate", "pre_tax_net_profit_rate",
        "after_tax_net_profit_rate", "continuous_profit_rate_after_tax",
        "operating_expense_rate", "research_development_expense_rate",
        "gross_profit_to_sales", "net_income_to_equity",
    ],
    "Liquidity": [
        "current_ratio", "quick_ratio",
        "working_capital_to_total_assets", "quick_asset_to_total_asset",
        "current_assets_to_total_assets", "cash_to_total_assets",
        "quick_asset_to_current_liabilities", "cash_to_current_liability",
        "current_liability_to_assets", "no_credit_interval",
        "cash_flow_rate",
    ],
    "Leverage & Debt": [
        "total_debt_to_net_value", "debt_ratio", "net_value_to_assets",
        "borrowing_dependency", "liability_to_equity", "equity_to_liability",
        "interest_bearing_debt_ratio", "interest_expense_ratio",
        "interest_coverage_ratio", "degree_of_financial_leverage",
        "liabilities_exceed_assets_flag", "long_term_fund_suitability_a",
    ],
    "Cash Flow": [
        "cash_flow_per_share", "cash_flow_to_sales",
        "cash_flow_to_total_assets", "cash_flow_to_liability",
        "cfo_to_assets", "cash_flow_to_equity",
        "operating_funds_to_liability", "cash_reinvestment_pct",
    ],
    "Growth": [
        "realized_sales_gross_profit_growth", "operating_profit_growth",
        "after_tax_net_profit_growth", "regular_net_profit_growth",
        "continuous_net_profit_growth", "total_assets_growth",
        "net_value_growth", "total_asset_return_growth",
    ],
    "Efficiency & Turnover": [
        "total_asset_turnover", "accounts_receivable_turnover",
        "average_collection_days", "inventory_turnover",
        "fixed_assets_turnover", "net_value_turnover",
        "current_assets_turnover_rate", "quick_assets_turnover_rate",
        "working_capital_turnover_rate", "cash_turnover_rate",
        "total_income_to_total_expense", "total_expense_to_assets",
    ],
    "Per Share & Per Employee": [
        "eps_last_four_quarters", "net_value_per_share_a",
        "net_value_per_share_b", "net_value_per_share_c",
        "revenue_per_share", "operating_profit_per_share",
        "pre_tax_net_profit_per_share",
        "revenue_per_employee", "operating_profit_per_employee",
        "allocation_rate_per_employee",
    ],
    "Balance Sheet": [
        "working_capital_to_equity", "current_liability_to_equity",
        "current_liability_to_total_liability", "long_term_liability_to_current_assets",
        "retained_earnings_to_total_assets", "net_income_to_total_assets",
        "total_assets_to_gnp_price", "fixed_assets_to_total_assets",
        "inventory_to_working_capital", "inventory_to_current_liability",
        "inventory_and_receivables_to_net_value",
        "net_income_negative_two_years_flag",
    ],
}

# Critical features: must be provided for a valid prediction
CRITICAL_FEATURES = [
    "roa_before_tax_c",
    "roa_after_tax_a",
    "operating_gross_margin",
    "current_ratio",
    "quick_ratio",
    "debt_ratio",
    "net_value_to_assets",
    "total_asset_turnover",
    "cash_flow_to_total_assets",
    "retained_earnings_to_total_assets",
    "working_capital_to_total_assets",
    "eps_last_four_quarters",
]

# All feature names in the order the model expects (excluding target column)
ALL_FEATURES = [v for k, v in COLUMN_MAPPING.items() if v != "bankruptcy_flag"]
