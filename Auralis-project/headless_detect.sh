ultralytics predict task=detect model=yolov5s.pt source=0 device=cpu imgsz=384 | \
grep "labels" | \
awk -F"labels" '{print $2}' | \
while read -r line; do
  espeak-ng "I see $line"
done

