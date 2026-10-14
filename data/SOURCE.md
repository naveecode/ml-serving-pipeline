# Telco Customer Churn

Public IBM sample, 7,043 customers and 21 columns.

SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

Retrieved 2026-10-14 from:

https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Upstream project: https://github.com/IBM/telco-customer-churn-on-icp4d

`TotalCharges` is blank for 11 customers with tenure 0. The loader coerces those cells to null and the model imputes them as 0.
