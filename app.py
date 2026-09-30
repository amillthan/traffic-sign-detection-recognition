import streamlit as st
from PIL import Image, ImageDraw

from src.pipeline import detect_and_classify

# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)


# ==================================================
# Application Header
# ==================================================

st.title("🚦 Traffic Sign Recognition System")

st.write(
    "Upload a road image and the system will automatically detect "
    "traffic signs using YOLO and classify each detected sign using "
    "the trained CNN model."
)

st.divider()


# ==================================================
# Image Upload
# ==================================================

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# Process Uploaded Image
# ==================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # --------------------------------------------------
    # Show Original Image
    # --------------------------------------------------

    st.subheader("📷 Uploaded Road Image")

    st.image(
        image,
        use_container_width=True
    )


    # --------------------------------------------------
    # Run YOLO + CNN Pipeline
    # --------------------------------------------------

    with st.spinner(
        "Detecting and classifying traffic signs..."
    ):

        results = detect_and_classify(
            image,
            detection_confidence=0.25,
            padding_ratio=0.15
        )


    # ==================================================
    # No Traffic Signs Detected
    # ==================================================

    if len(results) == 0:

        st.warning(
            "No supported traffic sign was detected in this image."
        )


    # ==================================================
    # Traffic Signs Detected
    # ==================================================

    else:

        st.success(
            f"{len(results)} traffic sign(s) detected."
        )


        # --------------------------------------------------
        # Create Annotated Image
        # --------------------------------------------------

        annotated_image = image.copy()

        draw = ImageDraw.Draw(annotated_image)


        # --------------------------------------------------
        # Draw Bounding Boxes and Labels
        # --------------------------------------------------

        for result in results:

            x1, y1, x2, y2 = result["bounding_box"]


            # Draw bounding box
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="red",
                width=4
            )


            # Create prediction label
            label = (
                f"{result['sign_name']} "
                f"{result['classification_confidence']:.1f}%"
            )


            # Calculate text dimensions
            text_bbox = draw.textbbox(
                (0, 0),
                label
            )

            text_width = (
                text_bbox[2] - text_bbox[0]
            )

            text_height = (
                text_bbox[3] - text_bbox[1]
            )


            # Position label above bounding box
            label_x = x1

            label_y = max(
                0,
                y1 - text_height - 10
            )


            # Prevent label from going outside
            # the right side of the image
            if label_x + text_width + 10 > image.width:

                label_x = max(
                    0,
                    image.width - text_width - 10
                )


            # Draw label background
            draw.rectangle(
                [
                    (label_x, label_y),
                    (
                        label_x + text_width + 10,
                        label_y + text_height + 8
                    )
                ],
                fill="red"
            )


            # Draw label text
            draw.text(
                (
                    label_x + 5,
                    label_y + 4
                ),
                label,
                fill="white"
            )


        # ==================================================
        # Detection Result
        # ==================================================

        st.subheader("🎯 Detection Result")

        st.image(
            annotated_image,
            use_container_width=True
        )


        # ==================================================
        # Detection Summary
        # ==================================================

        st.subheader("📊 Detection Summary")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Traffic Signs Detected",
                len(results)
            )

        with col2:

            average_detection_confidence = sum(
                result["detection_confidence"]
                for result in results
            ) / len(results)

            st.metric(
                "Average Detection Confidence",
                f"{average_detection_confidence:.2f}%"
            )


        # ==================================================
        # Individual Traffic Sign Results
        # ==================================================

        st.subheader("🔍 Detected Traffic Signs")


        for index, result in enumerate(
            results,
            start=1
        ):

            st.divider()

            st.markdown(
                f"### Traffic Sign {index}"
            )


            # Two-column layout
            col1, col2 = st.columns(
                [1, 2]
            )


            # --------------------------------------------------
            # Traffic Sign Crop
            # --------------------------------------------------

            with col1:

                st.image(
                    result["crop"],
                    caption="Detected Traffic Sign",
                    use_container_width=True
                )


            # --------------------------------------------------
            # Prediction Information
            # --------------------------------------------------

            with col2:

                st.markdown(
                    f"### {result['sign_name']}"
                )


                # CNN Confidence
                st.write(
                    "**CNN Classification Confidence**"
                )

                st.progress(
                    min(
                        result[
                            "classification_confidence"
                        ] / 100,
                        1.0
                    )
                )

                st.write(
                    f"{result['classification_confidence']:.2f}%"
                )


                # YOLO Detection Confidence
                st.write(
                    "**YOLO Detection Confidence**"
                )

                st.progress(
                    min(
                        result[
                            "detection_confidence"
                        ] / 100,
                        1.0
                    )
                )

                st.write(
                    f"{result['detection_confidence']:.2f}%"
                )


                # Bounding Box
                st.write(
                    "**Bounding Box:** "
                    f"{result['bounding_box']}"
                )


            # --------------------------------------------------
            # Top 3 CNN Predictions
            # --------------------------------------------------

            st.markdown(
                "#### Top 3 CNN Predictions"
            )


            for rank, prediction in enumerate(
                result["top_3"],
                start=1
            ):

                st.write(
                    f"**{rank}. "
                    f"{prediction['sign_name']}** "
                    f"— "
                    f"{prediction['confidence']:.2f}%"
                )


# ==================================================
# Initial Information
# ==================================================

else:

    st.info(
        "👆 Upload a JPG, JPEG, or PNG road image "
        "to begin traffic sign recognition."
    )


# ==================================================
# Footer
# ==================================================

st.divider()

st.caption(
    "Traffic Sign Recognition System | "
    "YOLO Traffic Sign Detection + CNN Classification"
)