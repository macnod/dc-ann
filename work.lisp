(defun make-time-tracker ()
  (let ((filename "/tmp/dc-utilities-time-tracker.ts"))
    (unless (file-exists filename)
      (let ((ds-time (ds `(:map :created (:map :internal-real-time
                                               ,(get-internal-real-time)
                                               :unix-time
                                               ,(get-universal-time))
                                :start-times ,(make-hash-table :test 'equal)))))
        (create-directory (path-only filename))
        (freeze-n-spew ds-time filename)))
    filename))

(defun mark-time (tag &key any-thread)
  "Marks the current time with TAG, for the purpose of later retrieving elapsed time.  See the elapsed-time function."
  (let* ((filename (make-time-tracker))
         (ds-time (slurp-n-thaw filename))
         (mutex (make-mutex :name "mark-time"))
         (mark (get-internal-real-time))
         (mark-name (if any-thread
                        tag
                        (format nil "~a-~a"
                            (sb-thread:thread-name sb-thread:*current-thread*)
                            tag))))
    (with-mutex (mutex)
      (ds-set ds-time `(:start-times ,mark-name) mark)
      (freeze-n-spew ds-time filename))
    mark))

(defun elapsed-time (tag &key any-thread)
  "Computes time elapsed since calling mark-time with TAG."
  (let* ((ds-time (slurp-n-thaw (make-time-tracker)))
         (mutex (make-mutex :name "elapsed-time"))
         (mark-name (if any-thread
                        tag
                        (format nil "~a-~a"
                            (sb-thread:thread-name sb-thread:*current-thread*)
                            tag))))
    (/ (- (get-internal-real-time)
          (with-mutex (mutex)
            (ds-get ds-time :start-times mark-name)))
       (float internal-time-units-per-second))))
