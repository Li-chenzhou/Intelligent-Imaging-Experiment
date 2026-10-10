import unittest,tempfile
from pathlib import Path
from stereo_experiment import np,cv2,read_pfm,metrics,matcher

class StereoTests(unittest.TestCase):
    def test_pfm_orientation_and_endian(self):
        a=np.array([[1.,2.,3.],[4.,np.inf,6.]],dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            for dtype,scale in [('<f4',-1),('>f4',1),('<f4',-.003922)]:
                p=Path(folder)/'test.pfm'
                with p.open('wb') as f:
                    f.write(f'Pf\n3 2\n{scale}\n'.encode());f.write(np.flipud(a).astype(dtype).tobytes())
                np.testing.assert_array_equal(read_pfm(p),a)
    def test_invalid_prediction_counted_as_failure(self):
        gt=np.array([[10,10,np.inf,10]],dtype='float32');pred=np.array([[10,-1,20,13]],dtype='float32')
        r=metrics(pred,gt,{'doffs':0,'baseline':100,'cam0':[[100,0,0],[0,100,0],[0,0,1]]})
        self.assertAlmostEqual(r['bad2_total'],200/3);self.assertAlmostEqual(r['coverage'],200/3);self.assertAlmostEqual(r['epe_valid'],1.5)
    def test_known_horizontal_disparity(self):
        rng=np.random.default_rng(7);left=rng.integers(0,256,(80,256),dtype=np.uint8);right=np.zeros_like(left)
        right[:,:-8]=left[:,8:]
        for kind in ['BM','SGBM']:
            raw=matcher(kind,9,32,32).compute(left,right).astype('float32')/16
            self.assertLess(abs(float(np.median(raw[12:-12,48:-16]))-8),.2)
    def test_half_scale_units(self):
        a=np.full((2,3),4,dtype=np.float32)
        b=cv2.resize(a,(6,4),interpolation=cv2.INTER_NEAREST)/.5
        self.assertTrue(np.all(b==8))

if __name__=='__main__':unittest.main()
