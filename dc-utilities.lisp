;; Copyright © 2003-2013 Donnie Cameron

;; Stuff that should be a part of Common Lisp. These routines are
;; general enough to be needed in most of my programs.

(defpackage :dc-utilities
  (:use :cl :cl-ppcre :postmodern :sb-thread :sb-ext)
  (:import-from :ironclad
                :ascii-string-to-byte-array
                :byte-array-to-hex-string
                :digest-sequence
                :sha512)
  (:export 
   
   λ
   λx
   ->
   ~>
   alist-values
   bytes-to-uint
   change-per-second
   command-line-options
   create-directory
   cull-named-params
   db-cmd
   directory-exists
   distinct-elements
   ds
   ds-clone
   ds-get
   ds-keys
   ds-list
   ds-merge
   ds-set
   ds-type
   factorial
   fast-compress
   fast-decompress
   fib
   file-exists
   file-line-count
   filter-file
   flatten
   freeze
   freeze-n-spew
   hash-keys
   hash-string
   hash-values
   home-settings-file
   index-values
   interrupt-sleep
   interruptible-sleep
   join-paths
   k-combination
   load-settings
   lof
   log-entry
   memoize
   memoize-with-limit
   range
   read-settings-file
   replace-regexs
   scrape-string
   sequence-bytes-to-uint
   sequence-uint-to-bytes
   setting
   setting
   shell-execute
   shift
   shuffle
   slurp
   slurp-n-thaw
   spew
   split-n-trim
   store-delete
   store-fetch
   store-path
   store-save
   temp-file-name
   thaw
   thread-pool-job-queue
   thread-pool-progress
   thread-pool-run-time
   thread-pool-start
   thread-pool-start-time
   thread-pool-stop
   thread-pool-stop-time
   thread-pool-time-to-go
   time-to-go
   timestamp
   to-ascii
   trim
   uint-to-bytes
   verify-string
   with-lines-in-file
   ))

(in-package :dc-utilities)

(set-dispatch-macro-character #\# #\%
  (lambda (s c n)
    (declare (ignore c n))
    (let ((list (read s nil (values) t)))
      (when (consp list)
        (ds (cons :map list))))))

(defparameter *interruptible-sleep-hash* (make-hash-table :test #'equal))
(defparameter *settings* nil)
(defparameter *dc-job-queue-mutex* nil)
(defparameter *dc-progress-mutex* nil)
(defparameter *dc-job-queue* nil)
(defparameter *dc-thread-pool-progress* nil)
(defparameter *dc-thread-pool-start-time* nil)
(defparameter *dc-thread-pool-stop-time* nil)

(defun to-ascii (s)
  (if (atom s)
      (map 'string (lambda (c) (if (> (char-code c) 127) #\Space c)) s)
      (loop for a in s collect
           (if (stringp a)
               (map 'string (lambda (c) (if (> (char-code c) 127) #\Space c)) a)
               (format nil "~a" a)))))

(defun timestamp (&key 
                    (time (get-universal-time))
                    string
                    (format "Y-M-DTh:m:s"))
  "Returns the give time (or the current time) formatted according to
   the format parameter, followed by an optional string.  If a string
   is provided, the function adds a space to the result and then
   appends the string to that.  The format string can contain any
   characters.  This function will replace the format characters Y, M,
   D, h, m, and s, respectively, with numbers representing the year,
   month, day, hour, minute, and second.  All the numbers are 2 digits
   long, except for the year, which is 4 digits long."
  (multiple-value-bind (second minute hour day month year)
      (decode-universal-time time)
    (let* ((space-string (if string (format nil " ~a" string) ""))
           (parts (ds (list :map 
                            "Y" (format nil "~d"     year)
                            "M" (format nil "~2,'0d" month)
                            "D" (format nil "~2,'0d" day)
                            "h" (format nil "~2,'0d" hour)
                            "m" (format nil "~2,'0d" minute)
                            "s" (format nil "~2,'0d" second))))
           (format-1 (loop for c across format
                        for s = (string c)
                        for d = (ds-get parts s)
                        collect (if d d s))))
      (concatenate 'string (format nil "~{~a~}" format-1) space-string))))

(defun log-entry (&rest messages)
  (timestamp :string (format nil "~{~a~}~%" messages)))

(defun replace-regexs (text list-of-regex-replacement-pairs &key ignore-case)
  (let ((ttext text))
    (loop for rp in list-of-regex-replacement-pairs
       do (setf ttext (ppcre:regex-replace-all
                       (format nil "~a~a" (if ignore-case "(?i)" "") (car rp))
                       ttext
                       (cdr rp))))
   ttext))

(defun scrape-string (regex string &key ignore-case)
  (map 'list 'identity
       (multiple-value-bind (whole parts)
           (ppcre:scan-to-strings
            (if ignore-case (concatenate 'string "(?is)" regex) regex)
            string)
         (declare (ignore whole))
         parts)))

(defun verify-string (string regexp)
  (multiple-value-bind (a b) (scan regexp string)
    (and a b (zerop a) (= b (length string)))))

(defun shell-execute (program &optional (parameters nil) (input-pipe-data ""))
  "Run shell program and return the output of the program as a string."
  (let ((parameters (cond ((null parameters) nil)
                          ((atom parameters) (list parameters))
                          (t parameters))))
    (with-output-to-string (output-stream)
      (with-input-from-string (input-stream input-pipe-data)
        (sb-ext:run-program program parameters :search t
                            :output output-stream :input input-stream)))))

(defun file-line-count (filename)
  "Obtain a count of the lines in the given file using the wc program."
  (values (parse-integer
           (shell-execute "wc" `("-l" ,filename)) :junk-allowed t)))

(defmacro with-lines-in-file ((line filename) &body body)
  "Runs body for each line in file filename"
  (let ((file (gensym)))
    `(with-open-file (,file ,filename)
      (do ((,line (read-line ,file nil) (read-line ,file nil)))
          ((null ,line) nil)
        ,@body))))

(defun join-paths (&rest path-parts)
  "Joins elements of path-parts into a file path, inserting slashes
   where necessary."
  (let ((path (format nil "~{~a~^/~}" 
                      (loop for part in path-parts collect
                           (regex-replace-all "^/|/$" part "")))))
    (format nil "~a~a"
            (if (verify-string (car path-parts) "^/.*$") "/" "")
            path)))

(defun create-directory (dir &key with-parents)
  (unless (directory-exists dir)
    (when 
        (zerop 
         (length 
          (shell-execute 
           "mkdir" (if with-parents (list "-p" dir) (list dir)))))
    dir)))

(defmacro filter-file ((line input-filename output-filename) &body body)
  "Copies lines from input to output, omitting lines for which body
   returns nil."
  (let ((output (gensym))
        (transformed-line (gensym)))
    `(with-open-file (,output ,output-filename
                              :direction :output
                              :if-exists :supersede)
       (with-lines-in-file (,line ,input-filename)
         (let ((,transformed-line ,@body))
           (when ,transformed-line (write-line ,transformed-line ,output)))))))

(defun freeze (object)
  "Serializes an object into a string, returning the string."
  (with-output-to-string (s) (write object :stream s :readably t)))

(defun thaw (string)
  "Deserializes an object from its string representation, returning
   the object."
  (with-input-from-string (s string) (read s)))

(defun slurp (filename)
  "Reads a whole file and returns the data of the file as a string."
  (with-open-file (stream filename)
    (let ((seq (make-array (file-length stream) 
                           :element-type 'character :fill-pointer t)))
      (handler-bind ((sb-int:stream-decoding-error
                      (lambda (c) 
                        (declare (ignore c))
                        (invoke-restart 'sb-int:attempt-resync))))
        (setf (fill-pointer seq) (read-sequence seq stream)))
      seq)))

(defun spew (string filename &key create-directories append)
  "Writes the contents of string to the file filename."
  (when create-directories
    (create-directory (directory-namestring filename) :with-parents t))
  (with-open-file (stream filename :direction :output :if-exists :supersede)
    (write-string string stream)))

(defun slurp-n-thaw (filename)
  "Reads and brings to life objects from a file."
  (with-open-file (stream filename) (read stream)))

(defun freeze-n-spew (object filename)
  "Writes the given object to a file."
  (with-open-file (stream filename :direction :output :if-exists :supersede)
    (write object :stream stream :readably t)))

(defun lof (filename)
  "Returns the length of the given file."
  (with-open-file (f filename) (file-length f)))

(defun temp-file-name (temp-folder)
  "Return a made-up, unique file name."
  (join-paths temp-folder (format nil "~a.tmp" (gensym))))

(defun split-n-trim (splitter-regex string)
  (remove-if (lambda (s) (zerop (length s)))
             (mapcar #'trim (split splitter-regex string))))

(defun trim (s)
  (regex-replace-all "^\\s+|\\s+$" s ""))

(defun flatten (l)
  (cond
    ((null l) nil)
    ((atom l) (list l))
    (t (loop for i in l append (flatten i)))))

(defun command-line-options (short-long-keyword-list)
  (loop for v in (cdr sb-ext:*posix-argv*)
     collect
       (loop named slk-loop for slk in short-long-keyword-list 
          when (member v slk :test 'equal) do (return-from slk-loop (third slk))
          finally (return-from slk-loop v))))

(defun shuffle (seq)
  (loop
     with l = (length seq) with w = (make-array l :initial-contents seq)
     for i from 0 below l for r = (random l) for h = (aref w i)
     do (setf (aref w i) (aref w r)) (setf (aref w r) h)
     finally (return (if (listp seq) (map 'list 'identity w) w))))

(defun memoize (f)
  (let ((cache (make-hash-table :test 'equal))
        (g (symbol-function f)))
    (setf (symbol-function f)
          (lambda (&rest p)
            (let ((v (gethash p cache)))
              (if v v (setf (gethash p cache)
                            (apply g p))))))))

(defun memoize-with-limit (f limit)
  (let ((cache (make-hash-table :test 'equal :size limit))
        (fifo nil)
        (g (symbol-function f)))
    (setf (symbol-function f)
          (lambda (&rest p)
            (let ((v (gethash p cache)))
              (if v v (progn
                        (when (>= (length fifo) limit)
                          (remhash (shift fifo) cache)
                          (nbutlast fifo))
                        (setf (gethash p cache) (apply g p)))))))))

(defun shift (list)
  (let ((value (car (last list))))
    (nbutlast list)
    value))

(defun fib (x)
  (cond
    ((zerop x) 0)
    ((= x 1) 1)
    (t (+ (fib (1- x)) (fib (- x 2))))))

(defun alist-values (alist &rest keys)
  (loop for key in keys collect (cdr (assoc key alist))))

(defun cull-named-params (named-params cull-keys)
  (let ((cull-keys (if (listp cull-keys) cull-keys (list cull-keys))))
    (loop for key in
         (remove-if (lambda (x) (member x cull-keys))
                    (loop for i from 0 below (length named-params) by 2
                       collect (elt named-params i)))
         appending (list key (getf named-params key)))))

(defun hash-keys (hash)
  (loop for a being the hash-keys in hash collect a))

(defun hash-values (hash)
  (loop for a being the hash-values in hash collect a))

(defun interruptible-sleep (secs name)
  (let ((target (+ (get-universal-time) secs)))
    (setf (gethash name *interruptible-sleep-hash*) nil)
    (loop while (and (< (get-universal-time) target)
                     (not (gethash name *interruptible-sleep-hash*)))
       do (sleep 1))
    (remhash name *interruptible-sleep-hash*)))

(defun interrupt-sleep (name)
  (setf (gethash name *interruptible-sleep-hash*) t))

(defun ds (list-or-atom &optional type)
  (let ((l (if (and type (listp list-or-atom) (not (null list-or-atom)))
               (cons type list-or-atom)
               list-or-atom)))
    (if (atom l)
        l
        (let ((type (pop l)))
          (case type
            (:map (loop with h = (make-hash-table :test #'equal)
                     while l
                     for key = (pop l)
                     for val = (ds (pop l))
                     do (setf (gethash key h) val)
                     finally (return h)))
            (:array (apply #'vector (mapcar 'ds l)))
            (:list (mapcar #'ds l))
            (t (error (format nil "Unknown collection type ~a" type))))))))

(defun ds-get (ds &rest keys)
  (if keys
      (case (ds-type ds)
        (hash-table
         (multiple-value-bind (value exists)
             (gethash (car keys) ds)
           (if exists
               (if (= (length keys) 1)
                   (values value t)
                   (values (apply #'ds-get (cons value (cdr keys))) t))
               (values nil nil))))
        (sequence
         (if (< (car keys) (length ds))
             (if (= (length keys) 1)
                 (values (elt ds (car keys)) t)
                 (values (apply #'ds-get (cons (elt ds (car keys))
                                               (cdr keys)))
                         t))
             (values nil nil)))
        (t (values nil nil)))
      (values ds t)))

(defun read-settings-file (&rest filepaths)
    (loop
       with settings-ds = (loop for filepath in filepaths
                             collect (ds (slurp-n-thaw filepath)))
       with settings = (car settings-ds)
       for ds in (cdr settings-ds)
       do (setf settings (ds-merge settings ds))
       finally (return settings)))

(defun load-settings (&rest filepaths)
  (setf *settings*
        (apply #'read-settings-file filepaths)))

(defun setting (&rest keys) (apply #'ds-get (cons *settings* keys)))

(defun ds-keys (ds &optional parent-keys)
  (case (ds-type ds)
    (hash-table 
     (loop for k being the hash-keys in ds 
        for new-parent-keys = (append parent-keys (list k))
        for child-ds = (gethash k ds)
        for child-keys = (ds-keys child-ds new-parent-keys)
        append child-keys))
    (sequence
     (loop for i from 0 below (length ds)
        for new-parent-keys = (append parent-keys (list i))
        for child-ds = (elt ds i)
        append (ds-keys child-ds new-parent-keys)))
    (t (list parent-keys))))

(defun ds-type (ds)
  (let* ((a (type-of ds))
         (b (string-downcase (format nil "~a" a))))
    (cond ((ppcre:scan 
            "simple-array character|vector character"
            b)
           'string)
          ((or (string= b "cons")
               (ppcre:scan "vector|array" b))
           'sequence)
          ((atom a) a)
          (t (car a)))))

(defun ds-set (ds location-key-path value)
  (let* ((keys (if (atom location-key-path)
                   (list location-key-path)
                   location-key-path))
        (key (car keys)))
    (if (= (length keys) 1)
        (progn
          (case (ds-type ds)
            (hash-table (setf (gethash key ds) value))
            (sequence (setf (elt ds key) value))
            (t (setf ds (make-hash-table)) (ds-set ds key value)))
          ds)
        (multiple-value-bind (target-ds exists)
            (ds-get ds key)
          (if exists
              (ds-set target-ds (cdr keys) value)
              (progn
                (case (ds-type ds)
                  (hash-table (setf (gethash key ds) (make-hash-table)))
                  (sequence (setf (elt ds key) (make-hash-table))))
                (setf target-ds (ds-get ds key))
                (ds-set target-ds (cdr keys) value)))))))

(defun ds-merge (ds-base &rest ds-set)
  (loop with ds-main = (ds-clone ds-base)
     for ds in ds-set
     do (loop for key-path in (ds-keys ds)
           do (ds-set ds-main key-path (apply #'ds-get (cons ds key-path))))
     finally (return ds-main)))

(defun ds-clone (ds)
  (case (ds-type ds)
    (hash-table
     (loop with ds-new = (make-hash-table :test 'equal)
        for key being the hash-keys in ds
          do (setf (gethash key ds-new) (ds-clone (gethash key ds)))
          finally (return ds-new)))
    (string
     (copy-seq ds))
    (sequence
     (if (equal (type-of ds) 'cons)
         (loop
            with ds-new = nil
            for i from 0 below (length ds)
            do (push (ds-clone (elt ds i)) ds-new)
            finally (return ds-new))
         (loop
            with l = (length ds)
            with ds-new = (make-array l)
            for i from 0 below l
            do (setf (elt ds-new i) (ds-clone (elt ds i)))
            finally (return ds-new))))
    (t ds)))
       
(defun ds-list (ds)
  (case (ds-type ds)
    (hash-table
     (loop with list = (list :map)
        for k being the hash-keys in ds
        for v = (gethash k ds)
        do (push k list)
          (push (ds-list v) list)
        finally (return (nreverse list))))
    (string
     (map 'string 'identity (copy-seq ds)))
    (sequence
     (if (equal (type-of ds) 'cons)
         (loop 
            with list = (list :list)
            for a in ds
            do (push (ds-list a) list)
            finally (return (nreverse list)))
         (loop
            with list = (list :array)
            for a across ds
            do (push (ds-list a) list)
            finally (return (nreverse list)))))
    (otherwise ds)))

(defun hash-string (password)
  "Hash a password and return a hex representation of the hash"
  (ironclad:byte-array-to-hex-string
   (ironclad:digest-sequence
    'ironclad:sha512
    (ironclad:ascii-string-to-byte-array (to-ascii password)))))

(defmacro λ (variables body)
  (let ((v (if (listp variables) variables (list variables))))
    `(lambda ,v ,body)))

(defmacro λx (body)
  (let* ((it (gensym))
         (xbody (subst it :x body)))
    (if (equal xbody body)
        `(lambda () ,xbody)
        `(lambda (,it) ,xbody))))

(defmacro -> (variables body)
  (let ((v (if (listp variables) variables (list variables))))
    `(lambda ,v ,body)))

(defmacro ~> (body)
  (let* ((it (gensym))
         (xbody (subst it :x body)))
    (if (equal xbody body)
        `(lambda () ,xbody)
        `(lambda (,it) ,xbody))))

(defun index-values (l)
  (loop with hash = (make-hash-table :test 'equal)
     for value in l
     for key = 0 then (1+ key)
     do (setf (gethash key hash) value)
     finally (return hash)))

(defun distinct-elements (list)
           (loop with h = (make-hash-table :test 'equal)
              for v = 0 then (1+ v)
              for k in list do (unless (gethash k h)
                                 (setf (gethash k h) v))
              finally
                (return (loop for u being the hash-keys of h
                           collect u into distinct-list
                           finally
                             (return (sort distinct-list
                                           (lambda (a b)
                                             (< (gethash a h) (gethash b h)))))))))

(defmacro db-cmd (ds db-command &rest params)
  "This function executes a database command.  The ds parameter is a
ds data structure with the following self-describing key value pairs:
:db, :username, :password, :host, :retry-count, :retry-sleep,
:retry-sleep-factor, and :log-function.  The :retry-sleep value is an
integer that represents the number seconds to sleep after the first
try.  Subsequent tries multiply the last amount of sleep time by the
value of :retry-sleep-factor.  The log function you provide must
accept a single string as a parameter.  All the parameters are
required."
  `(loop with log = (ds-get ,ds :log-function)
      for retries-left = (1- (ds-get ,ds :retry-count)) then (1- retries-left)
      for retry-sleep = (ds-get ,ds :retry-sleep) then
        (* retry-sleep (ds-get ,ds :retry-sleep-factor))
      for result =
        (handler-case
            (postmodern:with-connection
                (append
                 (mapcar (lambda (x) (ds-get ,ds x))
                         '(:db :username :password :host))
                 (list :pooled-p t))
              (,(intern (symbol-name db-command) "POSTMODERN") ,@params))
          (error (e)
            (funcall 
             log 
             (format nil "Error in database connection during try ~d"
                     (- (ds-get ,ds :retry-count) retries-left)))
            (funcall log (format nil "~a" e))
            :error))
      while (and (equal result :error) (not (zerop retries-left)))
      do
        (funcall log (format nil "Retrying in ~a seconds" retry-sleep))
        (sleep retry-sleep)
      finally (return result)))

(defun range (start end &key (step 1) (filter #'identity) random)
  (let ((range (loop for a from start to end by step
                  when (funcall filter a) collect a)))
    (if random (shuffle range) range)))

(defun change-per-second (function-or-symbol &optional (seconds 1))
  (let ((v1 (if (functionp function-or-symbol)
                (funcall function-or-symbol)
                (symbol-value function-or-symbol)))
        (v2 (progn (sleep seconds)
                   (if (functionp function-or-symbol)
                       (funcall function-or-symbol)
                       (symbol-value function-or-symbol)))))
    (/ (abs (- v1 v2)) (float seconds))))

(defun time-to-go (change-per-second record-count)
  (let* ((seconds (/ record-count (float change-per-second)))
         (hours (/ seconds 3600.0))
         (days (/ hours 24.0)))
    (list :@ change-per-second :to-go record-count
          :seconds seconds :hours hours :days days)))

(defun thread-pool-time-to-go (pool-name &optional total-record-count)
  (time-to-go
   (change-per-second (~> (thread-pool-progress pool-name)) 10)
   (- total-record-count (thread-pool-progress pool-name))))

(defun thread-pool-start-time (pool-name)
  (getf *dc-thread-pool-start-time* pool-name))

(defun thread-pool-stop-time (pool-name)
  (getf *dc-thread-pool-stop-time* pool-name))

(defun thread-pool-run-time (pool-name)
  (let ((start (thread-pool-start-time pool-name))
        (stop (thread-pool-stop-time pool-name)))
    (if start (- (if stop stop (get-universal-time)) start) 0)))

(defun thread-pool-progress (pool-name)
  (getf *dc-thread-pool-progress* pool-name))

(defun job-queue (pool-name)
  (getf *dc-job-queue* pool-name))

(defun thread-pool-start 
    (pool-name thread-count job-queue fn-job &optional fn-finally)
  "Starts thread-count threads using pool-name to name the threads and
runs fn-job with those threads.  Each thread runs fn-job, which takes
no parameters, in a loop.  When all the threads are done, this
function checks fn-finally.  If the caller provides fn-finally, then
this function returns with the result of calling fn-finally.  If the
caller doesn't provide fn-finally, then the this function exists with
a sum of the return values of all the threads that ran."
  (setf (getf *dc-thread-pool-progress* pool-name) 0)
  (setf (getf *dc-progress-mutex* pool-name)
        (make-mutex :name (symbol-name pool-name)))
  (setf (getf *dc-job-queue-mutex* pool-name)
        (make-mutex :name (symbol-name pool-name)))
  (setf (getf *dc-thread-pool-start-time* pool-name)
        (get-universal-time))
  (setf (getf *dc-thread-pool-stop-time* pool-name) nil)
  (make-thread
   (lambda ()
     (let* ((get-job (if (eql (type-of job-queue) 'function)
                         job-queue
                         (progn
                           (setf (getf *dc-job-queue* pool-name)
                                 (copy-list job-queue))
                           (lambda ()
                             (with-mutex ((getf *dc-job-queue-mutex* pool-name))
                               (pop (getf *dc-job-queue* pool-name)))))))
            (threads
             (loop
                for a from 1 to thread-count
                for name = (format nil "~a-~3,'0d" pool-name a)
                collect
                  (make-thread
                   (lambda ()
                     (loop for job = (funcall get-job)
                        while job
                        do (funcall fn-job job)
                          (with-mutex 
                              ((getf *dc-progress-mutex* pool-name))
                            (incf (getf *dc-thread-pool-progress* pool-name)))
                        summing 1))
                   :name name))))
       (loop for thread in threads
          summing (or (sb-thread:join-thread thread) 0) into total
          finally (progn
                    (setf (getf *dc-thread-pool-stop-time* pool-name)
                          (get-universal-time))
                    (return (if fn-finally
                                (funcall fn-finally)
                                total))))))
   :name (format nil "~a-000" pool-name)))

(defun thread-pool-stop (pool-name)
  (loop for threads = (remove-if-not 
                       (lambda (x) 
                         (scan (format nil "^~a" pool-name)
                               (sb-thread:thread-name x)))
                       (sb-thread:list-all-threads))
     while threads do
       (loop for thread in threads
          do (sb-thread:destroy-thread thread))
       (sleep 3)
     finally (sb-thread:list-all-threads)))


;;
;; Math
;;

(defun factorial (n)
  (loop for a from n downto 1
        for b = a then (* b a)
        finally (return b)))

(defun k-combination (k n)
  (/ (factorial n)
     (* (factorial k) (factorial (- n k)))))

;;
;; dc-store
;;

(defun file-exists (path)
  (let ((path (probe-file path)))
    (and path
         (not (equal (file-namestring path) "")))))

(defun directory-exists (path)
  (let ((path (probe-file path)))
    (and path
         (not (equal (directory-namestring path) ""))
         (equal (file-namestring path) ""))))

(defun store-path (root filename)
  (loop for a from 0 below 5
     for b = (* a 3)
     collect (subseq filename b (+ b 3)) into folders
     finally (return (format nil "~a/~a" root (apply #'join-paths folders)))))

(defun store-save (root key object)
  (let* ((contents (freeze object))
         (path (store-path root key))
         (abs-filename (join-paths path key)))
    (spew contents abs-filename :create-directories t)
    key))

(defun store-fetch (root key)
  (let* ((path (store-path root key))
         (abs-filename (join-paths path key)))
    (when (file-exists abs-filename)
      (slurp-n-thaw abs-filename))))

(defun store-delete (root key)
  (let* ((path (store-path root key))
         (abs-filename (join-paths path key))
         (article (store-fetch root key)))
    (when article
      (delete-file abs-filename)
      article)))
    
(defun uint-to-bytes (i &optional (size 4))
  (loop with ff = 255
     for a = i then (ash a -8)
     for b from 1 to size
     collect (logand a ff)))

(defun bytes-to-uint (byte-list)
  (loop for a in byte-list
     for b from 0 below (length byte-list)
     summing (* a (expt 2 (* b 8)))))

(defun sequence-uint-to-bytes (sequence &optional (size 4))
  (if (vectorp sequence)
      (loop with result = (make-array (* size (length sequence)))
         for a across sequence
         for index from 0 below (length sequence)
         do (loop with bytes = (uint-to-bytes a size)
               for byte in bytes
               for byte-index from 0 below size
               do (setf (aref result (+ (* index size) byte-index)) byte))
         finally (return result))
      (loop for a in sequence
           appending (uint-to-bytes a size))))

(defun sequence-bytes-to-uint (sequence &optional (size 4))
  (if (vectorp sequence)
      (loop with result = (make-array (/ (length sequence) size))
         for source-index from 0 below (length sequence) by size
         for result-index from 0 below (length result)
         do (setf (aref result result-index)
                  (bytes-to-uint
                   (map 'list 'identity
                        (subseq sequence source-index (+ source-index size)))))
         finally (return result))
      (loop for index from 0 below (length sequence) by size
           collecting (bytes-to-uint (subseq sequence index (+ index size))))))

(defun fast-compress (v)
  (loop with c = nil
     with l = (length v)
     for i from 0 below l
     for n = (aref v i)
     do (if (zerop n)
            (let ((run (loop for j = i then (1+ j)
                          while (< j l)
                          for m = n then (aref v j)
                          while (zerop m)
                          counting m into run
                          finally (return (list m run)))))
              (push (- (second run)) c)
              (unless (zerop (first run)) (push (first run) c))
              (incf i (second run)))
            (push n c))
     finally (return (nreverse c))))

(defun fast-decompress (l)
  (map 'vector 'identity
       (if (listp l)
           (loop for n in l appending
                (if (< n 0) (loop for a from 1 to (- n) collect 0) (list n)))
           (loop for n across l appending
                (if (< n 0) (loop for a from 1 to (- n) collect 0) (list n))))))

