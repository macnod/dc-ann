;; Copyright © 2002 Donnie Cameron
;;
;; ANN stands for Artificial Neural Network. This is a simple
;; implementation of the standard backpropagation neural network.
;;

(in-package :dc-ann)

(defclass t-cx ()
  ((target :reader target :initarg :target :initform (error ":neuron required")
           :type t-neuron)
   (weight :accessor weight :initarg :weight :initform 1.0 :type float)
   (delta :accessor delta :initform 0.0 :type float))
  (:documentation "Describes a neural connection to TARGET neuron.  WEIGHT represents the strength of the connection and DELTA contains the last change in WEIGHT.  TARGET is required."))



(defclass t-neuron ()
  ((net :reader net :initarg :net :initform (error ":net required") :type t-net)
   (layer :reader layer :initarg :layer :initform (error ":layer required")
          :type integer)
   (biased :reader biased :initarg :biased :initform nil :type boolean)
   (id :accessor id :type integer)
   (input :accessor input :initform 0.0 :type float)
   (output :accessor output :initform 0.0 :type float)
   (err :accessor err :initform 0.0 :type float)
   (x-coor :accessor x-coor :initform 0.0 :type float)
   (y-coor :accessor y-coor :initform 0.0 :type float)
   (cxs :accessor cxs :initform nil :type list))
  (:documentation "Describes a neuron.  NET, required, is an object of type t-net that represents the neural network that this neuron is a part of.  LAYER, required, is an object of type t-layer that represents the neural network layer that this neuron belongs to.  If BIASED is true, the neuron will not have incoming connections.  ID is a distinct integer that identifies the neuron. X-COOR and Y-COOR allow this neuron to be placed in 2-dimentional space.  CXS contains the list of outgoing connections (of type t-cx) to other neurons."))

(defclass t-layer ()
  ((neuron-array :accessor neuron-array :type vector)
   (layer-index :reader layer-index :initarg :layer-index :initform
                (error ":layer-index required") :type integer)
   (layer-type :reader layer-type :initarg :layer-type :initform
               (error ":layer-type required") :type keyword)
   (neuron-count :accessor neuron-count :initarg :neuron-count
                 :initform (error ":neuron-count required") :type integer)
   (net :reader net :initarg :net :initform (error ":net required")
        :type t-net)) 
  (:documentation "Describes a neural network layer."))

(defmethod initialize-instance :after ((layer t-layer) &key)
  (let ((size (+ (neuron-count layer)
                 (if (equal (layer-type layer) :hidden) 1 0))))
    (setf (neuron-count layer) size)
    (setf (neuron-array layer)
          (make-array size :element-type 't-neuron :fill-pointer 0))
    (loop for a from 0 below size do
         (vector-push
          (make-instance 't-neuron
                         :layer layer
                         :biased (and (equal (layer-type layer) :hidden)
                                      (= a (1- size)))
                         :net (net layer))
          (neuron-array layer)))))

(defun bound-sigmoid (neuron)
  (declare (t-neuron neuron))
  (let ((input (input neuron)))
    (cond
      ((> input 50.0) 1.0)
      ((< input -50.0) 0.0)
      (t (/ 1.0 (1+ (exp (- input))))))))

(defun bound-sigmoid-derivative (neuron)
  (declare (t-neuron neuron))
  (* (output neuron) (- 1 (output neuron))))

(defun rectified-linear (neuron)
  (max 0 (input neuron)))

(defun rectified-linear-derivative (neuron)
  (let ((output (output neuron)))
    (if (< output 0) 0 output)))

(defclass t-net ()
  ((topology :reader topology :initarg :topology
             :initform (error ":topology required"))
   (learning-rate :reader learning-rate :initarg :learning-rate :initform 0.1)
   (momentum :reader momentum :initarg :momentum :initform 0.3)
   (wi :accessor wi :initform 0)
   (transfer-function :accessor transfer-function
                      :initarg :transfer-function
                      :initform #'bound-sigmoid)
   (transfer-derivative :accessor transfer-derivative
                        :initarg :transfer-derivative
                        :initform #'bound-sigmoid-derivative)
   (layers :accessor layers)
   (next-id :accessor next-id :initform 0))
  (:documentation "Describes a standard multilayer, fully-connected backpropagation neural network."))

(defmethod initialize-instance :after ((neuron t-neuron) &key)
  (setf (id neuron) (incf (next-id (net neuron)))))

(defmethod output-layer ((net t-net))
  (car (last (layers net))))

(defmethod outputs ((net t-net))
  (map 'vector 'output (neuron-array (output-layer net))))

(defmethod input-layer ((net t-net))
  (car (layers net)))

(defmethod transfer ((neuron t-neuron))
  (setf (output neuron)
        (if (eq (layer-type (layer neuron)) :input)
            (input neuron)
            (funcall (transfer-function (net neuron)) neuron)))
  (setf (input neuron) (if (biased neuron) 1.0 0.0))
  neuron)

(defmethod connect ((net t-net))
  (loop for layer in (butlast (layers net)) do
       (loop for neuron across (neuron-array layer)
          for neuron-index = 1 then (1+ neuron-index)
          do (setf (cxs neuron)
                   (loop for target across
                        (neuron-array
                         (elt (layers net) (1+ (layer-index layer))))
                        for target-index = 1 then (1+ target-index)
                        when (not (biased target))
                        collect
                        (make-instance
                         't-cx
                         :target target
                         :weight (+ 0.5 (/ (sin (incf (wi net))) 2.0)))))))
  net)

(defmethod initialize-instance :after ((net t-net) &key)
  (setf (layers net)
        (loop for layer-spec in (topology net)
           for layer-index = 0 then (1+ layer-index)
           collect
             (make-instance
              't-layer
              :layer-index layer-index
              :layer-type
              (cond ((zerop layer-index) :input)
                    ((= layer-index (1- (length (topology net)))) :output)
                    (t :hidden))
              :neuron-count (elt (topology net) layer-index)
              :net net)))
  (connect net))

(defgeneric fire (object)
  (:method ((neuron t-neuron))
    (transfer neuron)
    (loop for cx in (cxs neuron)
       do (incf (input (target cx)) (* (weight cx) (output neuron))))
    neuron)
  (:method ((layer t-layer))
    (loop for neuron across (neuron-array layer) do (fire neuron))
    layer)
  (:method ((net t-net))
    (loop for layer in (layers net) do (fire layer))
    net))

(defgeneric feed (t-net inputs)
  (:method ((net t-net) (inputs list))
    (feed net (map 'vector 'identity inputs)))
  (:method ((net t-net) (inputs vector))
    ;; Copy input values to input layer
    (loop for neuron across (neuron-array (input-layer net))
       for a = 0 then (1+ a)
       do (setf (input neuron) (aref inputs a)))
    ;; Feed forward
    (fire net)
    ;; Return a vector with the output value of each output-layer neuron
    (map 'vector 'output (neuron-array (output-layer net)))))

(defgeneric backprop (component)
  (:method ((neuron t-neuron))
    (let ((err (loop for cx in (cxs neuron)
                  summing (* (err (target cx)) (weight cx))))
          (source-output (output neuron))
          (learning-rate (learning-rate (net neuron)))
          (momentum (momentum (net neuron))))
      (setf (err neuron)
            (if (eq (layer-type (layer neuron)) :input)
                err
                (* err (funcall (transfer-derivative (net neuron)) neuron))))
      (loop for cx in (cxs neuron) do
           (setf (delta cx)
                 (+ (* (err (target cx)) source-output learning-rate)
                    (* (delta cx) momentum)))
           (incf (weight cx) (delta cx))))
    neuron)
  (:method ((layer t-layer))
    (loop for neuron across (neuron-array layer) do (backprop neuron))
    layer)
  (:method ((net t-net))
    (loop for layer in (reverse (butlast (layers net)))
       do (backprop layer))
    net))

(defmethod output-layer-error ((net t-net) (outputs vector))
  (sqrt (loop for neuron across (neuron-array (output-layer net))
           for neuron-index from 0 below (length outputs)
           summing (let ((err (- (aref outputs neuron-index) (output neuron))))
                     (setf (err neuron)
                           (* err (funcall (transfer-derivative net) neuron)))
                     (* err err)))))

(defmethod learn-vector ((inputs vector) (outputs vector) (net t-net))
  (feed net inputs)
  (let ((err (output-layer-error net outputs)))
    (backprop net)
    err))

(defun tset-list-to-tset-vectors (tset)
  "Converts something like 
     '((0 0) (1) (0 1) (0) (1 0) (0) (1 1) (1))
   into something like
     #((#(0 0) #(1)) (#(0 1) #(0)) (#(1 0) #(0)) (#(1 1) #(1)))"
  (map 'vector 'identity
       (loop for i from 0 below (length tset) by 2 collect
            (list (map 'vector 'identity (elt tset i))
                  (map 'vector 'identity (elt tset (1+ i)))))))

(defgeneric present-vectors (t-net t-set)
  (:method ((net t-net) (t-set list))
    (loop
       with vec = (tset-list-to-tset-vectors t-set)
       with max-error = 0.0
       with shuffled-vector-indices =
         (shuffle (loop for a from 0 below (length vec) collect a))
       for i in shuffled-vector-indices
       for presentation = (elt vec i)
       for vector-error = (learn-vector (first presentation)
                                        (second presentation)
                                        net)
       when (< max-error vector-error)
       do (setf max-error vector-error)
       finally (return max-error))))

(defgeneric randomize-weights (object &key)
  (:method ((neuron t-neuron) &key max min)
    (declare (float max min))
    (if (= max min)
        (loop for cx in (cxs neuron) do
             (setf (weight cx) max)
             (setf (delta cx) 0))
        (loop for cx in (cxs neuron) do
             (setf (weight cx) (+ (random (- max min)) min))
             (setf (delta cx) 0)))
    neuron)
  (:method ((layer t-layer) &key max min)
    (declare (float max min))
    (loop for neuron across (neuron-array layer)
       do (randomize-weights neuron :max max :min min))
    layer)
  (:method ((net t-net) &key (max 0.25) (min (- max)))
    (declare (float max min))
    (loop for layer in (layers net) do
         (randomize-weights layer :max max :min min))
    net))

(defgeneric anneal (object variance)
  (:method ((neuron t-neuron) (variance float))
    (loop for cx in (cxs neuron)
       do (setf (weight cx)
                (+ (weight cx)
                   (/ (* (weight cx) (random variance)) 2)))
       finally (return neuron)))
  (:method ((layer t-layer) (variance float))
    (loop for neuron across (neuron-array layer)
       do (anneal neuron variance)
       finally (return layer)))
  (:method ((net t-net) (variance float))
    (loop for layer in (layers net)
         do (anneal layer variance)
         finally (return net))))


(defun elapsed (start-time)
  (- (get-universal-time) start-time))

(defun default-report-function (&key elapsed iteration mse)
  (format t "~as i~a e~a~%" elapsed iteration mse))

(defun default-status-function (&key net status elapsed iteration mse)
  (declare (ignore net))
  (format t "~a ~as i~a e~a~%" status elapsed iteration mse))

(defgeneric train (t-net t-set &key)
  (:method ((net t-net)
            (t-set list)
            &key 
              (target-mse 0.08)
              (max-iterations 10000)
              (report-frequency 1000)
              (report-function #'default-report-function)
              (status-function #'default-status-function)
              (logger-function nil)
              (randomize-weights nil)
              (annealing nil)
              (rerandomize-weights nil))
    (declare (float target-mse)
             (integer max-iterations report-frequency)
             (function report-function status-function))
    (loop
       initially
         (when status-function
           (funcall status-function
                    :net net :status "learning" :elapsed 0 :iteration 0
                    :mse 1.0))
         (when randomize-weights
           (if (listp randomize-weights)
               (apply #'randomize-weights (cons net randomize-weights))
               (randomize-weights net)))
       with start-time = (get-universal-time)
       with last-report-time
       with last-anneal-iteration = 0
       with rf = (* (/ report-frequency 1000)
                    internal-time-units-per-second)
       for i from 1 to max-iterations
       for mse = 1.0 then (present-vectors net t-set)
       while (> mse target-mse)
       when (and (> i 1) (> mse 0.80) (or rerandomize-weights annealing)) do
         (when (and (> mse 0.999) rerandomize-weights)
           (if (listp randomize-weights)
               (apply #'randomize-weights (cons net randomize-weights))
               (randomize-weights net))
           (when logger-function
             (funcall logger-function "randomized weights")))
         (when (and annealing (> (- i last-anneal-iteration) annealing))
           (anneal net 0.1)
           (setf last-anneal-iteration i)
           (when logger-function
             (funcall logger-function "annealed weights")))
       when (and report-function
                 (or (not last-report-time)
                     (>= (- (get-internal-real-time) last-report-time) rf)))
       do (funcall report-function
                   :elapsed (elapsed start-time) :iteration i :mse mse)
         (setf last-report-time (get-internal-real-time))
       finally (let ((elapsed (elapsed start-time))
                     (status (if (> mse target-mse) "maxi" "target")))
                 (when report-function
                   (funcall report-function
                            :elapsed elapsed :iteration i :mse mse))
                 (when status-function
                   (funcall status-function
                            :net net
                            :status status
                            :elapsed elapsed
                            :iteration i
                            :mse mse))
                 (return (values (elapsed start-time) i mse status)))))
  (:documentation "This function uses the standard backprogation of error method to train the neural network on the given sample set within the given constraints. Training is achieved when the target error reaches a level that is equal to or below the given target-mse value, within the given number of iterations. The function returns t if training is achieved and nil otherwise. If training is not achieved, the caller can call again to train for additional iterations. If randomize-weights is set to true, then the function starts training from scratch. This function accepts callback parameters that allow the function to periodically report on the progress of training. t-net is a list of alternating input and output lists, where each input/output list pair represents a single training vector.  Here's an example for the exclusive-or problem:

    '((0 0) (1) (0 1) (0) (1 0) (0) (1 1) (1))"))

(defgeneric object-freeze (object stream)
  (:method ((net t-net) (s stream))
    (format s "~a~%~a ~a~%"
            (topology net) (learning-rate net) (momentum net))
    (loop for layer in (layers net) do (object-freeze layer s)))
  (:method ((layer t-layer) (s stream))
    (loop for neuron across (neuron-array layer) do (object-freeze neuron s)))
  (:method ((neuron t-neuron) (s stream))
    (format s "~{~{~a ~a~}~^ ~}~%"
            (loop for cx in (cxs neuron)
               collect (list (weight cx) (delta cx))))))

(defun ann-freeze (net)
  (with-output-to-string (s) (object-freeze net s)))

(defgeneric object-thaw (object stream)
  (:method ((net t-net) (s stream))
    (loop for layer in (layers net) do (object-thaw layer s))
    (feed net (make-array (neuron-count (input-layer net)) :initial-element 0))
    net)
  (:method ((layer t-layer) (s stream))
    (loop for neuron across (neuron-array layer) do (object-thaw neuron s)))
  (:method ((neuron t-neuron) (s stream))
    (loop for cx in (cxs neuron) do
         (setf (weight cx) (read s))
         (setf (delta cx) (read s)))))

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
    (declare (float width height h-margin v-margin))
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
    (declare (float y delta margin))
    (loop
       for neuron across (neuron-array layer)
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
    (loop for inputs = (pop test-set)
       for outputs = (pop test-set)
       while inputs do
         (incf total)
         (let ((targets (feed net inputs)))
           (when (loop for output = (pop outputs)
                    for target across targets
                    always (or (and (>= target 0.5) (>= output 0.5))
                               (and (< target 0.5) (< output 0.5))))
             (incf correct)))
       finally (return (/ (truncate (* (/ correct total) 10000)) 100.0)))))

(defun train-n-test (&key
                       (training-file
                        (home-based "common-lisp/dc-ann/circle-training-data.csv"))
                       (test-file
                        (home-based "common-lisp/dc-ann/circle-test-data.csv"))
                       (topology '(2 50 1))
                       (learning-rate 0.1)
                       (momentum 0.3)
                       (transfer-function #'bound-sigmoid)
                       (transfer-derivative #'bound-sigmoid-derivative)
                       (trained-ann-file
                        (home-based "common-lisp/dc-ann/circle-ann-frozen.dat"))
                       (target-mse 0.05)
                       (max-iterations 1000000)
                       (randomize-weights '(:min -0.5 :max 0.5))
                       (annealing nil))
  (let* ((ann (make-instance 't-net
                             :topology topology
                             :learning-rate learning-rate
                             :momentum momentum
                             :transfer-function transfer-function
                             :transfer-derivative transfer-derivative))
         (t-set (read-data ann training-file))
         (mark (unique-name))
         (mark-time mark))
    (let ((training-results (train ann t-set
                                   :target-mse 0.05
                                   :randomize-weights t
                                   :max-iterations 100000
                                   :annealing annealing)))
      (spew (ann-freeze ann) trained-ann-file)
      (list :trained-ann-file trained-ann-file
            :ann-description (list :topology topology
                                   :learning-rate learning-rate
                                   :momentum momentum
                                   :transfer-function transfer-function)
            :training-parameters (list :training-file training-file
                                       :test-file test-file
                                       :target-mse target-mse
                                       :max-iterations max-iterations
                                       :randomize-weights randomize-weights
                                       :annealing annealing)
            :training-results training-results
            :training-time (elapsed-time mark)
            :trained-ann-accuracy (evaluate-training 
                                   ann (read-data ann test-file))))))

(defparameter *model-results* nil)
(defparameter *model-results-mutex* nil)

(defun next-parameter-item (parameter index)
  (when parameter
    (if (atom parameter)
        parameter
        (elt parameter index))))

(defun try-models (&key
                     (path (home-based "common-lisp/dc-ann"))
                     (training-file "circle-training-data.csv")
                     (test-file "circle-test-data.csv")
                     (trained-ann-file "circle-ann-frozen.dat")
                     (results-file "circle-results.dat")
                     ids
                     topologies
                     learning-rates
                     momenti
                     transfer-functions
                     transfer-derivatives
                     target-mses
                     max-iterations-s
                     randomize-weights-s
                     annealings
                     (thread-count 2))
  (when *model-results-mutex*
    (return-from try-models  "try-models is already running"))
  (setf *model-results* nil)
  (setf *model-results-mutex*
        (make-mutex :name (format nil "model-results-~a" (unique-name))))
  (loop with training-file = (join-paths path training-file)
     with test-file = (join-paths path test-file)
     with extension = (file-extension trained-ann-file)
     with tafno = (replace-extension trained-ann-file "")
     with results-file = (join-paths path results-file)
     for id in ids
     for index = 0 then (1+ index)
     for topology = (next-parameter-item topologies index)
     for learning-rate = (next-parameter-item learning-rates index)
     for momentum = (next-parameter-item momenti index)
     for transfer-function = (next-parameter-item transfer-functions index)
     for transfer-derivative = (next-parameter-item transfer-derivatives index)
     for target-mse = (next-parameter-item target-mses index)
     for max-iterations = (next-parameter-item max-iterations-s index)
     for randomize-weights = (when randomize-weights-s
                               (if (or (atom randomize-weights-s)
                                       (member (car randomize-weights-s)
                                               '(:min :max)))
                                   randomize-weights-s
                                   (elt randomize-weights-s index)))
     for annealing = (next-parameter-item annealings index)
     for taf = (join-paths path (format nil "~a-~a.~a" tafno id extension))
     collect
       (loop with parameters = (list :training-file training-file
                                     :test-file test-file
                                     :topology topology
                                     :learning-rate learning-rate
                                     :momentum momentum
                                     :transfer-function transfer-function
                                     :transfer-derivative transfer-derivative
                                     :trained-ann-file taf
                                     :target-mse target-mse
                                     :max-iterations max-iterations
                                     :randomize-weights randomize-weights
                                     :annealing annealing)
          for (key value) on parameters by #'cddr
          unless (null value)
          appending (list key value))
     into job-queue
     finally
       (return (thread-pool-start 
                :try-models
                (min thread-count (length ids))
                job-queue
                (lambda (standard-output job)
                  (let* ((*standard-output* standard-output)
                         (result (apply #'train-n-test job)))
                    (with-mutex (*model-results-mutex*)
                      (push result *model-results*)
                      (freeze-n-spew *model-results* results-file)
                    nil))
                (lambda (standard-output)
                  (let* ((*standard-output* standard-output))
                    (setf *model-results-mutex* nil)
                    (format t "All done!~%"))))))))

(defun stop-try-models ()
  (thread-pool-stop :try-models)
  (setf *model-results-mutex* nil))

