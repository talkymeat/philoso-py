import unittest
import numpy as np

from sb_statfuncs import mean, mode, std, infage, nanage
from test_materials import col, nans, infs

# sorted(col): [1, 4, 8, 9, 11, 13, 19, 19, 30, 40, 50, 69]
# nans: [nan, nan, nan, nan]
# infs: [inf, -inf, inf, -inf]


class TestSBStatfuncs(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean(col), 22.75)

    def test_mean_w_nans(self):
        self.assertEqual(mean(col+nans), 22.75)
        
    def test_mean_w_infs(self):
        self.assertEqual(mean(col+infs), 22.75)
        
    def test_mean_w_nans_infs(self):
        self.assertEqual(mean(col+nans+infs), 22.75)
        
    def test_mean_all_nan(self):
        self.assertEqual(mean(nans), 0.0)
        
    def test_mean_all_infs(self):
        self.assertEqual(mean(infs), 0.0)
        
    def test_mean_all_nans_infs(self):
        self.assertEqual(mean(nans+infs), 0.0)

    def test_mode(self):
        self.assertEqual(mode(col), 19.0)

    def test_mode_bimodal(self):
        self.assertEqual(mode(col+[11.0]), 15.0)

    def test_mode_w_nans(self):
        self.assertEqual(mode(col+nans), 19.0)

    def test_mode_w_infs(self):
        self.assertEqual(mode(col+infs), 19.0)

    def test_mode_w_nans_infs(self):
        self.assertEqual(mode(col+nans+infs), 19.0)

    def test_mode_all_nan(self):
        self.assertEqual(mode(nans), 0.0)

    def test_mode_all_infs(self):
        self.assertEqual(mode(infs), 0.0)

    def test_mode_all_nans_infs(self):
        self.assertEqual(mode(nans+infs), 0.0)

    def test_std(self):  
        self.assertEqual(round(std(col), 3), 19.842)

    def test_std_w_nans(self):  
        self.assertEqual(round(std(col+nans), 3), 19.842)

    def test_std_w_infs(self):  
        self.assertEqual(round(std(col+infs), 3), 19.842)

    def test_std_w_nans_infs(self):  
        self.assertEqual(round(std(col+nans+infs), 3), 19.842)

    def test_std_all_nan(self):  
        self.assertEqual(std(nans), 0.0)

    def test_std_all_infs(self):  
        self.assertEqual(std(infs), 0.0)

    def test_std_all_nans_infs(self):  
        self.assertEqual(std(nans+infs), 0.0)
    
    def test_nanage(self):
        real_vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        for nans in range(7):
            for infs in range(7):
                for ninfs in range(7):
                    for reals in range(7):
                        if nans or infs or ninfs or reals:
                            vals = [np.nan]*nans 
                            vals += [np.inf]*infs 
                            vals += [-np.inf]*ninfs
                            vals += real_vals[:reals]
                            self.assertAlmostEqual(nanage(vals), nans/(nans+infs+ninfs+reals))

    def test_infage(self):  
        real_vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        for nan_ct in range(7):
            for inf_ct in range(7):
                for ninf_ct in range(7):
                    for real_ct in range(7):
                        if nan_ct or inf_ct or ninf_ct or real_ct:
                            vals = [np.nan]*nan_ct 
                            vals += [np.inf]*inf_ct 
                            vals += [-np.inf]*ninf_ct
                            vals += real_vals[:real_ct]
                            self.assertAlmostEqual(infage(vals), (inf_ct+ninf_ct)/(nan_ct+inf_ct+ninf_ct+real_ct))


if __name__ == '__main__':
    unittest.main()