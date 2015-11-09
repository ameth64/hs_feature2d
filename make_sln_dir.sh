proj_name=""

while [ $# -ne 0 ];do
  case $1 in
    --project)
	shift
    proj_name="$1"
    shift
    ;;
  *)
  shift
  ;;
  esac
done

mkdir -p ci_script
mkdir -p cmake
mkdir -p include
if [ "${proj_name}" != "" ];then
  cd include/
  mkdir -p ${proj_name}
  cd ../
fi
mkdir -p src
if [ "${proj_name}" != "" ];then
  cd src/
  mkdir -p ${proj_name}
  cd ../
fi
mkdir -p test_data
mkdir -p unit_test