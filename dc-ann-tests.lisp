(in-package :cl-user)
(require :dc-ann)
(require :prove)
(defpackage dc-ann-tests (:use :cl :prove))
(in-package :dc-ann-tests)

(defun round-to-decimal-count (x decimal-count)
  (float (/ (round (* x (expt 10 decimal-count))) (expt 10 decimal-count))))

(plan 26)

(loop for (input expected-output) in 
     '((1 0.7311) (-1 0.2689) (1.0e10 1.0) (-1.0e10 0.0) 
       (1.0e20 1.0) (-1.0e20 0.0) (1.0e-10 0.5) (-1.0e-10 0.5) 
       (2 0.8808) (3 0.9526) (4 0.9820) (5 0.9933) 
       (6 0.9975) (7 0.9991) (8 0.9997) (9 0.9999) 
       (10 1.0) (-2 0.1192) (-3 0.0474) (-4 0.0180)
       (-5 0.0067) (-6 0.0025) (-7 0.0009) (-8 0.0003) 
       (-9 0.0001) (-10 0.0))
       for output = (round-to-decimal-count (dc-ann::logistic input) 4)
       do (is output expected-output))

(finalize)
