while getopts m: flag
do
	case "${flag}" in
		m) MEM = ${OPTARG};;
		*) echo "Ivalid option: -$flag";;
	esac
done

echo $MEM
#git add .
#git commit -m "$memo"
#git push
