while getopts :m:f: opt; do
  case $opt in
    f)
	git add $OPTARG
;;
    m)
	git commit -m "$OPTARG"
	git push
	;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

