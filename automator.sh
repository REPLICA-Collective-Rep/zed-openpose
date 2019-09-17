#!/bin/bash

root_dir="./"
is_test=false
is_clearing=false
use_email=""
email_freq=3

while getopts "i:e:s:tch" option; do
  case $option in 
    h ) 
		echo "-s [source directory of SVO files]"
		echo "-c [clear log.txt file when the script begins]"
		echo "-t [create and test dummy SVO files]"
		echo "-e [email address for sending log.txt notifications]"
		echo "-i [email frequency, ie. send email every nth file]"
		exit 1
    ;;
    c ) 
		is_clearing=true
    ;;
  	s ) 
	
		root_dir=$(echo $OPTARG/  | sed s#//*#/#g)
    ;;

    t )
		is_test=true

	;;
    i )
		email_freq=$OPTARG

	;;
    e )
		use_email="$OPTARG"

	;;
  esac
done

dest_dir=$(echo "$root_dir/completed"  | sed s#//*#/#g)
err_dir=$(echo "$root_dir/failed"  | sed s#//*#/#g)
log_file=$(echo "$root_dir/log.txt"  | sed s#//*#/#g)

mkdir -p $dest_dir
mkdir -p $err_dir


function log {

	pattern="+[%d-%m-%y][%H:%M:%S]"
	if [[ $2 != "" ]]; then
		pattern=$2
	fi

	echo `date $pattern` $1 | tee -a $log_file
}


log "[i] source dir: $root_dir"
log "[i] destination dir: $dest_dir"
log "[i] failed dir: $err_dir"
log "[i] email address: $use_email"
log "[i] email frequency: every $email_freq files"


if $is_clearing ; then
	> $log_file
	log "[~] clearing $log_file"
fi
if $is_test ; then
	log "[~] generating testfiles..."
	for f in {1..10}
	do 
		touch $root_dir/testfile$f.svo
		log "[~] touch testfile$f.svo"
	done
fi

current_index=1
total_files=`ls -f $root_dir*.svo | wc -l  | xargs`

log "[i] processing $total_files files..."

function send_email {
	if [[ $use_email != "" ]]; then
		echo "sending email to $use_email ..."
		message="Extraction complete $current_index of $total_files"

		str_log=`cat $log_file`
		request="curlmail.co/$use_email?subject="
		# echo $str_log
		echo "$(<$log_file)" | curl -d @- $request
	fi
}

for f in $root_dir*.svo
do
	log "[$current_index/$total_files] opening $f"
	./build/zed_openpose -net_resolution 320x240 -model_pose MPI_4_layers  -svo_path $f
	mv $f $dest_dir

	if [[ $is_test ]]; then
		sleep 0.1
	fi

	if ! (($current_index % $email_freq)) && (( current_index != 0 )) ; then
		send_email  # using https://curlmail.co
	fi

	((current_index++))
done

send_email # using https://curlmail.co