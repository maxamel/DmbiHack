## declare an array variable
declare -a arr=("baby_monitor" "lights" "motion_sensor" "security_camera" "smoke_detector" "socket" "thermostat" "TV" "watch" "water_sensor")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   # or do whatever with individual element of the array
   awk -v var="$i" -F, '{ $298 = ($298 ~ var ? 1 : 0) } 1' OFS=, hackathon.csv > "$i".csv
done
