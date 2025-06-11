
/*
import delimited "C:\Users\sb3357.SPI-9VS5N34\Princeton Dropbox\Sam Barnett\inelastic-capital-data (1)\temp_files\main_sample_BN.csv", clear 
*/ 

import delimited "C:\Users\illge\Princeton Dropbox\Sam Barnett\inelastic-capital-data (1)\temp_files\main_sample_BN.csv", clear // seems like maybe a key sign change, etc missing? first check GDP issue. Theirs is [1973, 2011]


egen ind_gr = group(naics)
xtset ind_gr year, yearly

*gen FE_year_1973 = (year == 1973)
xi i.year, noomit

rename _I* FE_*

foreach var of varlist FE_year_* {
	replace `var' = `var' * lexp_share
}

codebook naics
keep if year >= 1973 & year <= 2011

/*
*Nonlinear OLS 
ivreg2 dln_pip dln_ip c.dln_ip#c.lutil_dm  ///
lutil_dm dln_cap c.dln_cap#c.lutil_dm dln_uvcip i.ind_gr i.year FE_year_*, dkraay(3) first partial(i.ind_gr i.year FE_year_*)

*All IVs 
ivreg2 dln_pip (dln_ip c.dln_ip#c.lutil_dm = dln_frgn_rgdp dln_m_shea_inst ///
	c.dln_frgn_rgdp#c.lutil_dm c.dln_m_shea_inst#c.lutil_dm) ///
	lutil_dm dln_cap c.dln_cap#c.lutil_dm dln_uvcip i.ind_gr i.year FE_year_*, dkraay(3) first partial(i.ind_gr i.year FE_year_*)
*/

*Try checking stuff against their data directly 
tostring naics, replace 
rename * sam_*
rename (sam_naics sam_year) (naics3 year)


preserve
*********************************** COPY PASTE FROM BOEHM ETAL CODE 
use "C:\Users\sb3357.SPI-9VS5N34\Princeton Dropbox\Sam Barnett\inelastic-capital-data (1)\raw\original\boem_pandalai-nayar_2022\empirics\main analysis\temp_files\data_main_sample", clear 

egen ind_gr = group(naics3)
xtset ind_gr year, yearly

*baseline is Dln_M_shea_inst2
gen Dln_M_shea_inst = Dln_M_shea_inst2

* replace missing prices with ppi for year 2012 for which price data is not available
replace Dln_Pip = Dln_bls_ppi if Dln_Pip == . & Dln_bls_ppi != . & year == 2012
replace Dln_P 	= Dln_bls_ppi if Dln_P == . & Dln_bls_ppi != . & year == 2012

* construct leads and lags before constraining sample to 1973 to 2011
gen FDln_Pip = F.Dln_Pip
gen FDln_P = F.Dln_P

gen LDln_ip = L.Dln_ip
gen LD_inv = L.D_inv

* ensure that sample is always identical
drop if year <= 1972
drop if year >= 2012
drop FE_year_2012 FE_year_2013 FE_year_2014 FE_year_2015 FE_year_2016
drop lag2_FE_year_2012 lag2_FE_year_2013 lag2_FE_year_2014 lag2_FE_year_2015 lag2_FE_year_2016

** winsorize all variables at p99 and p1
foreach var in Dln_P Dln_ship Dln_X Dln_vprod DD_inv Dln_UVC Dln_VC Dln_ip ///
 Dln_cap Lutil Lutil_dm Lutil_dm_perc Lutil_dm_samp D_G D_IM ///
 Dln_bls_ppi Dln_Pip Dln_UVCip Dln_n_firms Dln_n_estabs ///
 FDln_Pip FDln_P Dln_M_shea_inst Dln_frgn_rgdp Dln_frgn_rgdp_lag2 Dln_er {
 
	egen ub = pctile(`var') , p(99)
	egen lb = pctile(`var') , p(1)
	replace `var' = ub if `var' > ub & `var' != .
	replace `var' = lb if `var' < lb & `var' != .
	drop ub lb
	
}
* Note: the shocks in Dln_M_shea_inst, Dln_frgn_rgdp, and Dln_er should not be
* winsorized in a01_construct_sample.do

* construct summary variables to impose cross-coefficient restrictions for reduced form
				
gen Dln_quant2 = Dln_frgn_rgdp + Dln_M + Dln_qi_pce + Dln_qi_equip + Dln_qi_struct ///
				+ D_G 
				
gen Dln_price = Dln_frgn_gdp_defl + Dln_PM + Dln_pi_pce + Dln_pi_equip + Dln_pi_struct

********************************************************************************
** De-mean all variables entering interactions in sample
********************************************************************************
* see SW file demeaning in regressions with interaction terms.tex for details
* the utilization rate has to be demeaned in sample to obtain the correct interpretation of the slope coefficient
* Further Dln_ip has to be demeaned when testing the model prediction that the demeaned utilization rate has a coefficient of zero
* we therefore demean all variables entering interaction terms here
	
foreach var of varlist Dln_ip Dln_cap Lutil_dm Dln_frgn_rgdp Dln_frgn_rgdp_lag2 Dln_M_shea_inst Dln_er Dln_er_lag2 Lvinv_dm {

	sum `var', det
	qui reg `var' 
	qui predict `var'_dm, resid
	qui replace `var' = `var'_dm
	qui drop `var'_dm
	qui sum `var', det
	
}

* construct high utilization rate dummy
egen mean_util = mean(Lutil), by(naics3)
sum Lutil
gen util_high = (mean_util > r(mean))
drop mean_util

* construct interaction for first stage
gen Dln_ip_Lutil_dm = c.Dln_ip#c.Lutil_dm

* construct utilization bins for the nonparametric estimates
egen p85 = pctile(Lutil_dm) , p(85)
egen p50 = pctile(Lutil_dm) , p(50)
egen p15 = pctile(Lutil_dm) , p(15)

gen Lutil_bin = 1
replace Lutil_bin = 2 if Lutil_dm >= p15		
replace Lutil_bin = 3 if Lutil_dm >= p50		
replace Lutil_bin = 4 if Lutil_dm >= p85 	
drop p85 p50 p15
*********************************** COPY PASTE FROM BOEHM ETAL CODE 
tempfile BOEHM 
save `BOEHM'
restore 

merge 1:1 naics3 year using `BOEHM', keep(matched)

corr sam_dln_frgn_rgdp Dln_frgn_rgdp
tw (scatter sam_dln_frgn_rgdp Dln_frgn_rgdp) (line Dln_frgn_rgdp Dln_frgn_rgdp)

corr sam_dln_er Dln_er
tw (scatter sam_dln_er Dln_er) (line Dln_er Dln_er)

corr sam_dln_pip Dln_Pip
tw (scatter sam_dln_pip Dln_Pip) (line Dln_Pip Dln_Pip)

corr sam_dln_uvcip Dln_UVCip 
tw (scatter sam_dln_uvcip Dln_UVCip ) (line Dln_UVCip Dln_UVCip)

corr sam_dln_m_shea_inst Dln_M_shea_inst
tw (scatter sam_dln_m_shea_inst Dln_M_shea_inst) (line Dln_M_shea_inst Dln_M_shea_inst)

corr sam_lutil_dm Lutil_dm
tw (scatter sam_lutil_dm Lutil_dm ) (line Lutil_dm Lutil_dm)

corr sam_dln_cap Dln_cap 
tw (scatter sam_dln_cap Dln_cap ) (line Dln_cap Dln_cap)


