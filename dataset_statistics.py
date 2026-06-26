import os
import csv
import json
import statistics

CSV_FILE = "analysis/dataset.csv"
REPORT_FILE = "analysis/statistics.txt"
PROFILE_FILE = "analysis/crop_profile.json"

if not os.path.exists(CSV_FILE):
    print("dataset.csv not found")
    exit()

face_percent = []
face_width = []
face_height = []
eye_distance = []
mouth_distance = []
angle = []
brightness = []
blur = []

with open(CSV_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        try:

            face_percent.append(float(row["Face %"]))
            face_width.append(float(row["Face Width"]))
            face_height.append(float(row["Face Height"]))
            eye_distance.append(float(row["Eye Distance"]))
            mouth_distance.append(float(row["Mouth Distance"]))
            angle.append(float(row["Face Angle"]))
            brightness.append(float(row["Brightness"]))
            blur.append(float(row["Blur"]))

        except:
            pass


def stat(values):

    return {

        "min": round(min(values),2),
        "max": round(max(values),2),
        "avg": round(statistics.mean(values),2),
        "median": round(statistics.median(values),2),
        "std": round(statistics.pstdev(values),2)

    }


report = {

    "Face Percent": stat(face_percent),
    "Face Width": stat(face_width),
    "Face Height": stat(face_height),
    "Eye Distance": stat(eye_distance),
    "Mouth Distance": stat(mouth_distance),
    "Angle": stat(angle),
    "Brightness": stat(brightness),
    "Blur": stat(blur)

}


# -----------------------------------------
# Recommended Crop Parameters
# -----------------------------------------

avg_face_percent = report["Face Percent"]["avg"]

avg_eye = report["Eye Distance"]["avg"]

profile = {

    "target_output_width":300,
    "target_output_height":400,

    "ideal_face_percent":avg_face_percent,

    "ideal_eye_distance":avg_eye,

    "eye_position_percent":38,

    "top_margin_percent":18,

    "bottom_margin_percent":27,

    "left_margin_percent":18,

    "right_margin_percent":18

}


with open(PROFILE_FILE,"w",encoding="utf-8") as f:

    json.dump(profile,f,indent=4)


with open(REPORT_FILE,"w",encoding="utf-8") as f:

    f.write("="*60+"\n")
    f.write("PhotoAI Dataset Statistics\n")
    f.write("="*60+"\n\n")

    for key,value in report.items():

        f.write(f"{key}\n")

        for k,v in value.items():

            f.write(f"   {k:10} : {v}\n")

        f.write("\n")

    f.write("="*60+"\n")
    f.write("Recommended Crop Profile\n")
    f.write("="*60+"\n\n")

    for k,v in profile.items():

        f.write(f"{k:30} : {v}\n")


print()

print("="*60)
print("Statistics Generated")
print("="*60)

print()

for k,v in report.items():

    print(k)

    print(v)

    print()

print("="*60)
print("Crop Profile Saved")
print("="*60)