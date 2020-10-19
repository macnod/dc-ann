; Copyright © 2002 Donnie Cameron
;;
;; ANN stands for Artificial Neural Network. This is a simple
;; implementation of the standard backpropagation neural network.
;;

(in-package :dc-ann)

(defclass t-transfer-function ()
  ((name :reader name :initarg :name :initform (error ":name required"))
   (transfer :reader transfer :initarg :transfer :initform (error ":transfer required"))
   (derivative :reader derivative :initarg :derivative :initform (error ":derivative required"))))

(defparameter *transfer-functions* (make-hash-table))

(setf (gethash :logistic *transfer-functions*)
      (make-instance 't-transfer-function 
                     :name :logistic
                     :transfer (lambda (x) (/ 1.0 (1+ (exp (- x)))))
                     :derivative (lambda (x) (* x (- 1 x)))))

(setf (gethash :relu *transfer-functions*)
      (make-instance 't-transfer-function 
                     :name :relu
                     :transfer (lambda (x) (max 0 x))
                     :derivative (lambda (x) (if (> x 0.0) 1.0 0.0))))

(defclass t-cx ()
  ((target :reader target :initarg :target :initform (error ":neuron required")
           :type t-neuron)
   (weight :accessor weight :initarg :weight :initform 1.0 :type real))
  (:documentation "Describes a neural connection to TARGET neuron. WEIGHT represents the strength of the connection and DELTA contains the last change in WEIGHT. TARGET is required."))

(defclass t-neuron ()
  ((net :reader net :initarg :net :initform (error ":net required") :type t-net)
   (layer :accessor layer :initarg :layer :initform (error ":layer required"))
   (layer-type :accessor layer-type :initarg :layer-type :initform (error ":layer-type required"))
   (biased :reader biased :initarg :biased :initform nil :type boolean)
   (id :accessor id)
   (input :accessor input :initform 0.0 :type real)
   (transfer-function :accessor transfer-function :initarg :transfer-function 
                      :initform (gethash :logistic *transfer-functions*))
   (output :accessor output :initform 0.0 :type real)
   (expected-output :accessor expected-output :initform 0.0 :type real)
   (err :accessor err :initform 0.0 :type real)
   (delta :accessor delta :initform 0.0 :type real)
   (derivative :accessor derivative :initform 0.0 :type real)
   (x-coor :accessor x-coor :initform 0.0 :type real)
   (y-coor :accessor y-coor :initform 0.0 :type real)
   (cxs :accessor cxs :initform nil :type list))
  (:documentation "Describes a neuron.  NET, required, is an object of type t-net that represents the neural network that this neuron is a part of.  LAYER, required, is an object of type t-layer that represents the neural network layer that this neuron belongs to.  If BIASED is true, the neuron will not have incoming connections.  ID is a distinct integer that identifies the neuron. X-COOR and Y-COOR allow this neuron to be placed in 2-dimentional space.  CXS contains the list of outgoing connections (of type t-cx) to other neurons."))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (setf (id neuron) 
        (format nil "~d-~d" 
                (layer-index (layer neuron)) 
                (incf (neuron-index (layer neuron))))))

(defclass t-layer ()
  ((neurons :accessor neurons :type list :initform nil)
   (transfer-tag :accessor transfer-tag :initarg :transfer-tag :initform :logistic)
   (layer-index :reader layer-index :initarg :layer-index :initform
                (error ":layer-index required") :type integer)
   (layer-type :reader layer-type :initarg :layer-type :initform
               (error ":layer-type required") :type keyword)
   (neuron-count :accessor neuron-count :initarg :neuron-count
                 :initform (error ":neuron-count required") :type integer)
   (neuron-index :accessor neuron-index :initform 0 :type integer)
   (next-layer :accessor next-layer :initarg :next-layer
               :type 't-layer :initform nil)
   (net :reader net :initarg :net :initform (error ":net required")
        :type t-net))
  (:documentation "Describes a neural network layer."))

(defmethod initialize-instance :after ((layer t-layer) &key)
  (when (equal (layer-type layer) :hidden)
    (incf (neuron-count layer)))
  (setf (neurons layer)
        (loop with neuron-count = (neuron-count layer)
           for a from 0 below neuron-count
           for is-biased = (and (= a (1- neuron-count))
                                (equal (layer-type layer) :hidden))
           collect (make-instance 't-neuron
                                  :layer layer
                                  :layer-type (layer-type layer)
                                  :biased is-biased
                                  :net (net layer)
                                  :transfer-function (gethash (transfer-tag layer)
                                                              *transfer-functions*)))))

(defgeneric set-transfer-function (target function-name)
  (:method ((neuron t-neuron) function-name)
    (setf (transfer-function neuron) (gethash function-name *transfer-functions*)))
  (:method ((layer t-layer) function-name)
    (setf (transfer-tag layer) function-name)))

(defclass t-net ()
  ((topology :reader topology :initarg :topology
             :initform (error ":topology required"))
   (learning-rate :accessor learning-rate :initarg :learning-rate :initform 0.3)
   (momentum :accessor momentum :initarg :momentum :initform 0.8)
   (wi :accessor wi :initform 0.0)
   (layers :accessor layers)
   (next-id :accessor next-id :initform 0)
   (min-mse :accessor min-mse :type real :initform 1000000.0)
   (max-mse :accessor max-mse :type real :initform -1000000.0)
   (mse-list :accessor mse-list :type list :initform nil)
   (id :reader id :initarg :id :type string :initform (unique-name))
   (log-file :accessor log-file :type string :initform nil)
   (last-anneal-iteration :accessor last-anneal-iteration
                          :type integer
                          :initform 0)
   (randomize :accessor randomize :type boolean :initform nil)
   (anneal :accessor anneal :type boolean :initform nil)
   (stop-training :accessor stop-training :type boolean :initform nil)
   (shock :accessor shock :type boolean :initform nil)
   (state :reader state :initform (make-random-state)))
  (:documentation "Describes a standard multilayer, fully-connected backpropagation neural network."))

(defmethod output-layer ((net t-net))
  (car (last (layers net))))

(defmethod input-layer ((net t-net))
  (car (layers net)))

(defmethod outputs ((net t-net))
  (mapcar #'output (neurons (output-layer net))))

(defmethod set-expected-outputs (net outputs)
  (loop 
     for neuron in (neurons (output-layer net))
     for value in outputs
     do (setf (expected-output neuron) value)))

(defmethod expected-outputs ((net t-net))
  (mapcar #'expected-output (neurons (output-layer net))))

(defmethod inputs ((net t-net))
  (mapcar #'input (neurons (input-layer net))))

(defmethod hidden-layers ((net t-net))
  (butlast (cdr (layers net))))

(defmethod transfer ((neuron t-neuron))
  (setf (output neuron)
        (funcall (transfer (transfer-function neuron)) (input neuron)))
  (setf (input neuron) 0.0)
  neuron)

(defmethod initialize-instance :after ((net t-net) &key)
  (setf (layers net)
        (loop with layer-count = (length (topology net))
           for layer-spec in (topology net)
           for layer-neuron-count = (getf layer-spec :count)
           for layer-transfer-tag = (getf layer-spec :transfer)
           for layer-index = 0 then (1+ layer-index)
           for layer-type = (cond ((zerop layer-index) :input)
                                  ((= layer-index (1- layer-count)) :output)
                                  (t :hidden))
           collect
             (make-instance
              't-layer
              :layer-index layer-index
              :layer-type layer-type
              :neuron-count layer-neuron-count
              :transfer-tag layer-transfer-tag
              :net net)))
  (loop for layer in (butlast (layers net))
     for next-layer in (cdr (layers net))
     do (setf (next-layer layer) next-layer))
  (connect net))


(defmethod connect ((net t-net))
  (loop for layer in (butlast (layers net)) do
       (loop for neuron in (neurons layer)
          do (setf (cxs neuron)
                   (loop for target in (neurons (next-layer layer))
                      when (not (biased target))
                      collect (make-instance 't-cx :target target :weight 0.0)))))
  net)

(defgeneric set-inputs (object inputs)
  (:method ((layer t-layer) (inputs list))
    (loop for neuron in (neurons layer)
       for input in inputs
       collect (setf (input neuron) input)))
  (:method ((net t-net) (inputs list))
    (loop for neuron in (neurons (input-layer net))
       for input in inputs
       collect (setf (input neuron) input))))

(defun layer-inputs (layer)
  (mapcar #'input (neurons layer)))

(defun layer-outputs (layer)
  (mapcar #'output (neurons layer)))

(defgeneric list-neurons (object)
  (:method ((layer t-layer))
    (neurons layer))
  (:method ((net t-net))
    (loop for layer in (layers net) appending (neurons layer))))

(defun neuron-by-id (net id)
  (loop for neuron in (list-neurons net)
     when (equal (id neuron) id)
     return neuron))

(defun neuron-by-xy (net layer-index neuron-index) 
  (elt (neurons (elt (layers net) layer-index)) neuron-index))

(defgeneric fire (object)
  (:method ((neuron t-neuron))
    (transfer neuron)
    (loop for cx in (cxs neuron)
       do (incf (input (target cx)) (* (weight cx) (output neuron))))
    neuron)
  (:method ((layer t-layer))
    (loop for neuron in (neurons layer) do (fire neuron))
    layer)
  (:method ((net t-net))
    (loop for layer in (layers net) do (fire layer))
    net))

(defmethod feed ((net t-net) (input-row list))
  (loop
     for input in input-row
     for neuron in (neurons (input-layer net))
     do (setf (input neuron) input))
  (fire net)
  (outputs net))

(defmethod feed-multiple ((net t-net) (input-rows list))
  (loop for input-row in input-rows
     collect (feed net input-row)))

(defmethod compute-neuron-error ((neuron t-neuron))
  (setf (err neuron)
        (if (equal (layer-type neuron) :output)
            (- (expected-output neuron) (output neuron))
            (loop for cx in (cxs neuron)
               summing (* (weight cx) (err (target cx))))))
  (setf (delta neuron)
        (* (err neuron) (funcall (derivative (transfer-function neuron)) (output neuron)))))

(defmethod update-neuron-weights ((neuron t-neuron))
  (loop for cx in (cxs neuron)
     for delta = (* (learning-rate (net neuron))
                    (delta neuron)
                    (input neuron))
     do (incf (weight cx) delta)))

(defmethod backprop (net)
  (loop for layer in (reverse (layers net)) do
       (loop for neuron in (neurons layer) do
            (compute-neuron-error neuron)
            (update-neuron-weights neuron))))

(defmethod network-error (net)
  (sqrt (loop for neuron in (neurons (output-layer net))
           summing (expt (err neuron) 2))))

(defmethod learn-vector ((net t-net) (inputs list) (outputs list))
  (set-expected-outputs net outputs)
  (feed net inputs)
  (backprop net)
  (network-error net))

(defgeneric present-vectors (t-net t-set)
  (:method ((net t-net) (t-set list))
    (loop
       for training-vector in t-set
       for inputs = (car training-vector)
       for outputs = (cadr training-vector)
       for vector-error = (learn-vector net inputs outputs)
       collect vector-error into vector-error-collection
       finally (return (/ (apply '+ vector-error-collection)
                          (float (length vector-error-collection)))))))

(defgeneric randomize-weights (object &key)
  (:method ((neuron t-neuron) &key max min)
    (declare (real max min))
    (if (= max min)
        (loop for cx in (cxs neuron) do
             (setf (weight cx) max))
        (loop for cx in (cxs neuron) do
             (setf (weight cx) (+ (random (- max min) (state (net neuron))) min))))
      neuron)
  (:method ((layer t-layer) &key max min)
    (declare (real max min))
    (loop for neuron in (neurons layer)
       do (randomize-weights neuron :max max :min min))
    layer)
  (:method ((net t-net) &key (max 0.5) (min 0.0))
    (declare (real max min))
    (loop for layer in (layers net) do
         (randomize-weights layer :max max :min min))
    net))

(defgeneric anneal-weights (object variance)
  (:method ((neuron t-neuron) (variance real))
    (loop for cx in (cxs neuron)
       do (setf (weight cx)
                (+ (weight cx)
                   (/ (* (weight cx) (random variance (state (net neuron)))) 2)))
       finally (return neuron)))
  (:method ((layer t-layer) (variance real))
    (loop for neuron in (neurons layer)
       do (anneal-weights neuron variance)
       finally (return layer)))
  (:method ((net t-net) (variance real))
    (loop for layer in (layers net)
         do (anneal-weights layer variance)
         finally (return net))))

(defun elapsed (start-time)
  (- (get-universal-time) start-time))

(defun default-report-function (&key net elapsed iteration mse)
  (when (> mse (max-mse net)) (setf (max-mse net) mse))
  (when (< mse (min-mse net)) (setf (min-mse net) mse))
  (push (list iteration mse) (mse-list net))
  (with-open-file (stream (log-file net) :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (write-log-entry stream
                     (format nil "~a ~a ~5$ ~4$ ~4$"
                             elapsed iteration
                             mse (min-mse net) (max-mse net)))))

(defun default-status-function (&key net status elapsed iteration mse)
  (when (> mse (max-mse net)) (setf (max-mse net) mse))
  (when (< mse (min-mse net)) (setf (min-mse net) mse))
  (push (list iteration mse) (mse-list net))
  (with-open-file (stream (log-file net) :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (write-log-entry stream
                     (format nil "~a ~a ~a ~5$ ~4$ ~4$"
                             status elapsed iteration
                             mse (min-mse net) (max-mse net)))))

(defun default-logger-function (net message)
  (with-open-file (stream (log-file net) :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (write-log-entry stream message)))

(defun initialize-training (net
                            log-file
                            status-function
                            randomize-weights)
  (setf (log-file net)
        (if log-file
            log-file
            (format nil "/tmp/training-~a.log" (id net))))
  (setf (max-mse net) -1000000.0)
  (setf (min-mse net) 1000000.0)
  (setf (last-anneal-iteration net) 0)
  (setf (anneal net) nil)
  (setf (randomize net) nil)
  (setf (stop-training net) nil)
  (setf (shock net) nil)
  (when status-function
    (funcall status-function
             :net net
             :status "learning"
             :elapsed 0
             :iteration 0
             :mse 1.0))
  (when randomize-weights
    (if (listp randomize-weights)
        (apply #'randomize-weights (cons net randomize-weights))
        (randomize-weights net))
    (setf (mse-list net) nil)))

(defun shock-weights (net
                      target-mse
                      mse
                      rerandomize-weights
                      randomize-weights
                      logger-function
                      annealing
                      i)
  (when (or (and (> mse (* target-mse 10))
                 rerandomize-weights)
            (shock net))
    (if (listp randomize-weights)
        (apply #'randomize-weights (cons net randomize-weights))
        (randomize-weights net))
    (when logger-function
      (funcall logger-function net "randomized weights")))
  (when (and annealing
             (or (shock net)
                 (> (- i (last-anneal-iteration net))
                    annealing)))
    (anneal-weights net 0.1)
    (setf (last-anneal-iteration net) i)
    (when logger-function
      (funcall logger-function "annealed weights")))
  (setf (randomize net) nil)
  (setf (anneal net) nil)
  (setf (shock net) nil))

(defgeneric train (t-net t-set &key)
  (:method ((net t-net)
            (t-set list)
            &key
              (target-mse 0.08)
              (max-iterations 1000000)
              (report-frequency 1000)
              (report-function #'default-report-function)
              (status-function #'default-status-function)
              (logger-function #'default-logger-function)
              (randomize-weights '(:max 0.5 :min 0.0))
              (annealing nil)
              (rerandomize-weights nil)
              (log-file nil))
    (declare (real target-mse)
             (integer max-iterations report-frequency)
             (function report-function status-function))
    (make-thread
     (lambda ()
       (loop
          initially (initialize-training net
                                         log-file
                                         status-function
                                         randomize-weights)
          with start-time = (get-universal-time)
          with last-report-time
          with rf = (* (/ report-frequency 1000)
                       internal-time-units-per-second)
          for i from 1 to max-iterations
          for mse = 1.0 then (present-vectors net t-set)
          while (and (> mse target-mse) (not (stop-training net)))
          when (or
                (and (> i 1)
                     (> mse (* 10 target-mse))
                     (or rerandomize-weights annealing))
                (randomize net)
                (anneal net)
                (shock net))
          do (shock-weights net target-mse mse rerandomize-weights
                            randomize-weights logger-function annealing i)
          when (and report-function
                    (or (not last-report-time)
                        (>= (- (get-internal-real-time) last-report-time) rf)))
          do (funcall report-function
                      :net net
                      :elapsed (elapsed start-time)
                      :iteration i
                      :mse mse)
            (setf last-report-time (get-internal-real-time))
          finally (let ((elapsed (elapsed start-time))
                        (status (if (> mse target-mse) "maxi" "target")))
                    (when report-function
                      (funcall report-function
                               :net net
                               :elapsed elapsed
                               :iteration i
                               :mse mse))
                    (when status-function
                      (funcall status-function
                               :net net
                               :status status
                               :elapsed elapsed
                               :iteration i
                               :mse mse))
                    (return (list :elapsed (elapsed start-time)
                                  :iterations i
                                  :error mse
                                  :status status)))))
     :name (format nil "training-~(~a~)" (id net))))
  (:documentation "This function uses the standard backprogation of error method to train the neural network on the given sample set within the given constraints. Training is achieved when the target error reaches a level that is equal to or below the given target-mse value, within the given number of iterations. The function returns t if training is achieved and nil otherwise. If training is not achieved, the caller can call again to train for additional iterations. If randomize-weights is set to true or to a value like '(:min -1.0 :max 1.0), which is the default, then the function starts training from scratch. Otherwise, if randomize-weights is set to nil, training resumes from where it left of in the last call.  This function accepts callback parameters that allow the function to periodically report on the progress of training. t-net is a list of alternating input and output lists, where each input/output list pair represents a single training vector.  Here's an example for the exclusive-or problem:

    '((0 0) (1) (0 1) (0) (1 0) (0) (1 1) (1))"))

(defgeneric object-freeze (object stream)
  (:method ((net t-net) (s stream))
    (format s "~a~%~a ~a~%"
            (topology net) (learning-rate net) (momentum net))
    (loop for layer in (layers net) do (object-freeze layer s)))
  (:method ((layer t-layer) (s stream))
    (loop for neuron in (neurons layer) do (object-freeze neuron s)))
  (:method ((neuron t-neuron) (s stream))
    (format s "~{~{~a ~a~}~^ ~}~%"
            (loop for cx in (cxs neuron)
               collect (weight cx)))))

(defun ann-freeze (net)
  (with-output-to-string (s) (object-freeze net s)))

(defgeneric object-thaw (object stream)
  (:method ((net t-net) (s stream))
    (loop for layer in (layers net) do (object-thaw layer s)))
  (:method ((layer t-layer) (s stream))
    (loop for neuron across (neuron-array layer) do (object-thaw neuron s)))
  (:method ((neuron t-neuron) (s stream))
    (loop for cx in (cxs neuron) do
         (setf (weight cx) (read s)))))

(defun ann-thaw (string)
    (with-input-from-string (stream string)
      (object-thaw (make-instance 't-net
                                  :topology (read stream)
                                  :learning-rate (read stream)
                                  :momentum (read stream))
                   stream)))

(defgeneric layout-neurons (t-net &key)
  (:method ((net t-net) &key
            (width 200.0)
            (height 100.0)
            (h-margin 20.0)
            (v-margin 20.0))
    (declare (real width height h-margin v-margin))
    (let* ((c-width (- width (* h-margin 2)))
           (v-delta (/ (- height (* v-margin 2))
                              (1- (length (layers net))))))
      (loop
         for layer in (layers net)
         for y = v-margin then (+ y v-delta)
         for h-delta = (/ c-width (neuron-count layer))
         do (layout-neurons layer :y y :delta h-delta :margin h-margin)))
    net)
  (:method ((layer t-layer) &key y delta margin)
    (declare (real y delta margin))
    (loop
       for neuron in (neurons layer)
       for x = (+ margin (/ delta 2)) then (+ x delta)
       do
         (setf (x-coor neuron) x)
         (setf (y-coor neuron) y))
    layer))

(defun read-data (net csv-file)
  (let (tset)
    (with-lines-in-file (row csv-file)
      (when (not (zerop (length (trim row))))
        (let* ((numbers (mapcar #'parse-number (split-n-trim row :on-regex ",")))
               (inputs (subseq numbers 0 (car (topology net))))
               (outputs (subseq numbers (car (topology net)))))
          (push outputs tset)
          (push inputs tset))))
    tset))

(defun evaluate-training (net test-set)
  (let ((total 0.0)
        (correct 0.0))
    (loop for test-row in test-set
       for inputs = (car test-set)
       for outputs = (cadr test-set)
       do 
         (incf total)
         (let ((targets (feed net inputs)))
           (when (loop for output in outputs
                    for target in targets
                    always (or (and (>= target 0.5) (>= output 0.5))
                               (and (< target 0.5) (< output 0.5))))
             (incf correct)))
       finally (return (/ (truncate (* (/ correct total) 10000)) 100.0)))))

(defun evaluate-training-1hs (net test-set &optional include-log)
    (loop
       for test in test-set
       for inputs = (car test)
       for expected-outputs = (cadr test)
       for expected-winner = (index-of-max expected-outputs)
       for outputs = (feed net inputs)
       for winner = (index-of-max outputs)
       for total = 1 then (1+ total)
       for pass = (= winner expected-winner)
       for correct = (if pass 1 0) then (if pass (1+ correct) correct)
       collect (list :outputs outputs 
                     :expected expected-outputs 
                     :pass (if pass "pass" "fail")) into log
       finally (return (let ((result (list :total total :correct correct)))
                         (if include-log (append result (list :log log))
                             result)))))
         
(defun xor-data ()
  "She drew a circle that shut me out. <p>Heretic rebel, a thing to flout.  But, love and I had the wit to win.  We drew a circle that took her in."  
  '(((0 0) (1))
    ((0 1) (0))
    ((1 0) (0))
    ((1 1) (1))))

(defun xor-data-1hs ()
  '(((0 0) (1 0))
    ((0 1) (0 1))
    ((1 0) (0 1))
    ((1 1) (1 0))))

(defun circle-data (net count)
  (loop with state = (state net)
     for a from 1 to count
     for x = (random 1.0 state)
     for y = (random 1.0 state)
     for r = (sqrt (+ (* x x) (* y y)))
     for z = (if (< r 0.707) 1.0 0.0)
     collect (list (list x y) (list z))))

(defun circle-data-1hs (net count)
  (loop with true = 0.0 and false = 0.0 and state = (state net)
     for a from 1 to count
     for x = (random 1.0 state)
     for y = (random 1.0 state)
     for r = (sqrt (+ (* x x) (* y y)))
     if (< r 0.707) do (setf true 1.0 false 0.0)
     else do (setf true 0.0 false 1.0)
     collect (list (list x y) (list false true))))

(defun shuffle (net seq)
  (loop
     with l = (length seq) with w = (make-array l :initial-contents seq)
     for i from 0 below l for r = (random l (state net)) for h = (aref w i)
     do (setf (aref w i) (aref w r)) (setf (aref w r) h)
     finally (return (if (listp seq) (map 'list 'identity w) w))))

