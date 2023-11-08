import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


def get_body_bounding_box(image):
    # MediaPipeのPoseモデルを初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # 画像からポーズを検出
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # キーポイントが検出されなければNoneを返す
    if not results.pose_landmarks:
        return None

    # 検出された全身のキーポイントからバウンディングボックスを計算
    landmark_coords = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
    x_coords, y_coords = zip(*landmark_coords)
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)

    # 画像の幅と高さを取得
    img_height, img_width, _ = image.shape

    # バウンディングボックスの座標をピクセル単位に変換
    bbox = [
        int(xmin * img_width),
        int(ymin * img_height),
        int(xmax * img_width),
        int(ymax * img_height),
    ]

    return bbox


def crop_image_with_margin(image, bbox, margin):
    # 画像の高さと幅を取得
    img_height, img_width, _ = image.shape

    # バウンディングボックスの座標に余白を加える
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(img_width, x_max + margin)
    y_max = min(img_height, y_max + margin)

    # バウンディングボックスに基づいて画像をクロップ
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image


def add_text_to_image(image, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    line_type = 2

    # テキストの背景色 (黒色)
    rectangle_bgr = (0, 0, 0)
    # テキストの下にある矩形の大きさを取得
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, line_type)
    # テキストを配置する矩形の座標を設定
    rectangle_coords = (
        (position[0], position[1] + 5),
        (position[0] + text_width + 5, position[1] - text_height - 5),
    )
    # 矩形を画像に描画
    cv2.rectangle(
        image, rectangle_coords[0], rectangle_coords[1], rectangle_bgr, cv2.FILLED
    )
    # 矩形の上にテキストを描画
    cv2.putText(
        image, text, (position[0], position[1]), font, font_scale, font_color, line_type
    )
    return image


def resize_images(image1, image2):
    # クロップされた画像の高さが異なる場合はリサイズして一致させる
    height1 = image1.shape[0]
    height2 = image2.shape[0]
    if height1 != height2:
        # 新しい高さは二つの画像の中で小さい方に合わせる
        new_height = min(height1, height2)
        image1 = cv2.resize(
            image1, (int(image1.shape[1] * new_height / height1), new_height),
        )
        image2 = cv2.resize(
            image2, (int(image2.shape[1] * new_height / height2), new_height),
        )
    return image1, image2


def combine_images(image1, image2):
    # 画像を横に並べる
    combined_img = np.hstack((image1, image2))

    return combined_img


def main():
    # Streamlitアプリのタイトル
    st.title("過去の自分と体系比較")

    # 画像アップロード部分
    uploaded_file1 = st.file_uploader("過去の画像を選択", type=["jpg", "png", "jpeg"])
    uploaded_file2 = st.file_uploader("現在の画像を選択", type=["jpg", "png", "jpeg"])

    # アップロードされた画像がある場合の処理
    if uploaded_file1 is not None and uploaded_file2 is not None:
        # PIL形式で画像を読み込む
        image1 = Image.open(uploaded_file1)
        image2 = Image.open(uploaded_file2)

        # OpenCV形式に変換する
        image1 = np.array(image1.convert("RGB"))
        image2 = np.array(image2.convert("RGB"))

        # バウンディングボックスを取得してクロップする
        bbox1 = get_body_bounding_box(image1)
        bbox2 = get_body_bounding_box(image2)
        mergin = 100
        if bbox1 and bbox2:
            cropped_img1 = crop_image_with_margin(image1, bbox1, mergin)
            cropped_img2 = crop_image_with_margin(image2, bbox2, mergin)
            cropped_img1, cropped_img2 = resize_images(cropped_img1, cropped_img2)

            # 文字の追加
            cropped_img1 = add_text_to_image(cropped_img1, "KAKO", (30, 50))
            cropped_img2 = add_text_to_image(cropped_img2, "IMA", (30, 50))

            combined_img = combine_images(cropped_img1, cropped_img2)

            # Streamlitで画像を表示
            st.image(combined_img, use_column_width=True)
        else:
            st.write("身体が上手く検出できませんでした。")


if __name__ == "__main__":
    main()

