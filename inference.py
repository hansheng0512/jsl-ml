from PIL import Image, ImageDraw, ImageFont
import sentencepiece as spm
import onnxruntime
import numpy as np
import torch
import textwrap
import math
import hashlib
import time

global batch_size
batch_size = 1

global patch_size
patch_size = {"height":16,"width":16}

global max_patches
max_patches=2048

global num_channels
num_channels = 3

global patch_elements
patch_elements = (patch_size["height"] * patch_size["width"] * num_channels ) + 2

global encoder_cache
encoder_cache = {}

global decoder_cache
decoder_cache = {}

def load_models(encoder_path,decoder_path,cuda):
    if cuda:
        provider = ["CUDAExecutionProvider"]
    else:
        provider = ['CPUExecutionProvider']

    encoder_session = onnxruntime.InferenceSession(encoder_path, providers = provider)
    #load decoder
    decoder_session = onnxruntime.InferenceSession(decoder_path, providers = provider)

    return encoder_session, decoder_session

def render_text(text, text_size=36, text_color="black", background_color="white",
                left_padding=5, right_padding=5, top_padding=5, bottom_padding=5,
                font_path=None):

    # Wrap the input text to fit within a specified width (80 characters in this case)
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=text)
    wrapped_text = "\n".join(lines)

    # Load the font to be used for the text
    font = ImageFont.truetype(font_path, encoding="UTF-8", size=text_size)

    # Create a temporary image to calculate the text dimensions
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), background_color))

    # Calculate the width and height of the wrapped text using the specified font
    _, _, text_width, text_height = temp_draw.textbbox((0, 0), wrapped_text, font)

    # Calculate the width and height of the final image, including padding
    image_width = text_width + left_padding + right_padding
    image_height = text_height + top_padding + bottom_padding

    # Create a new image with the calculated dimensions and background color
    image = Image.new("RGB", (image_width, image_height), background_color)

    # Create a drawing context for the image
    draw = ImageDraw.Draw(image)

    # Draw the wrapped text on the image at the specified coordinates with the specified font and colors
    draw.text(xy=(left_padding, top_padding), text=wrapped_text, fill=text_color, font=font)

    return image

def render_header(image, header, font_path=None):
    """
    Adds a header to the top of the image.
    The Header is having White Background and black text.
    """
    header_image = render_text(header, font_path=font_path)

    # Calculate the new width of the combined image, taking the maximum width of the header or the original image
    new_width = max(header_image.width, image.width)

    # Calculate the new height of the combined image, preserving the aspect ratio
    new_height = int(image.height * (new_width / image.width))

    # Calculate the height of the header image in the new combined image
    new_header_height = int(header_image.height * (new_width / header_image.width))

    # Create a new blank image with a white background that can accommodate both the header and the original image
    new_image = Image.new("RGB", (new_width, new_height + new_header_height), "white")

    # Paste the header image at the top of the new image
    new_image.paste(header_image.resize((new_width, new_header_height)), (0, 0))

    # Paste the original image below the header in the new image
    new_image.paste(image.resize((new_width, new_height)), (0, new_header_height))

    return new_image

def torch_extract_patches(image_tensor, patch_height, patch_width,log=True):
    """
    Extracts Patches from the image with header.
    """

    # Add a batch dimension to the image tensor
    image_tensor = image_tensor.unsqueeze(0)

    # Use the unfold operation to extract patches from the image tensor
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))

    # Reshape the patches tensor to have the desired dimensions
    patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)

    # Permute the dimensions to have the correct order
    # Reshape the patches tensor to the final shape
    patches = patches.permute(0, 4, 2, 3, 1).reshape(
        image_tensor.size(2) // patch_height,
        image_tensor.size(3) // patch_width,
        image_tensor.size(1) * patch_height * patch_width,
        )

    return patches.unsqueeze(0)

def extract_flattened_patches(image,log=True):
    """
    Extracts Patches.
    Converts the Patches into a format acceptable by Pix2Struct Encoder.
    """
    image = image.transpose((2, 0, 1))

    image = torch.from_numpy(image)
    patch_height, patch_width = patch_size["height"], patch_size["width"]
    image_height, image_width = image.shape[1], image.shape[2]

    scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
    num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
    num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
    resized_height = max(num_feasible_rows * patch_height, 1)
    resized_width = max(num_feasible_cols * patch_width, 1)

    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0).to(torch.float32)

    patches = torch_extract_patches(image, patch_height, patch_width)

    patches_shape = patches.shape
    rows = patches_shape[1]
    columns = patches_shape[2]
    depth = patches_shape[3]

    patches = patches.reshape([rows * columns, depth])

    row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
    col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

    row_ids += 1
    col_ids += 1

    row_ids = row_ids.to(torch.float32)
    col_ids = col_ids.to(torch.float32)

    result = torch.cat([row_ids, col_ids, patches], -1)

    result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

    result = result.numpy()

    return result

def normalize(image):
    """
    Normalize the Image using Mean and Standard Deviation
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)
    adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

    return (image - mean) / adjusted_stddev

def _attention_mask(flattened_patches):
    """
    Create Attention Mask from Final Flattened Patches.
    """
    attention_masks = [(flattened_patches.sum(axis=-1) != 0).astype(np.int64)]
    return attention_masks[0]

def preprocess_image(input_image, header_text=None, font_path=None, weights=32, log=True):
    """
    Wrapper Method for Flattened Patches and Encoder Attention Mask Creation.
    """
    # Load and preprocess the image
    image = Image.open(input_image)

    # Render header text if provided
    if header_text:
        image = render_header(image, header_text, font_path=font_path)

    patches = normalize(np.array(image))

    # Extract flattened patches
    patches = extract_flattened_patches(patches,log)

    if weights == 16:
        flattened_patches = np.around(patches, decimals=4).reshape((batch_size,max_patches,patch_elements)).astype(np.float16)
    else:
        flattened_patches = np.around(patches, decimals=4).reshape((batch_size,max_patches,patch_elements))

    attention_mask = _attention_mask(flattened_patches)

    return flattened_patches,attention_mask

def run_encoder(encoder_,flattened_patches,encoder_attention_mask):
    """
    Runs the encoder.
    """
    output = encoder_.run(None, {"flattened_patches": flattened_patches,
                                 "attention_mask": encoder_attention_mask})

    return output

def run_decoder(decoder_,encoder_attention_mask,ehs):
    """
    Runs the Decoder.
    No Caching Implementation
    """
    encoded_question_input = np.array([[0]])
    decoder_attention_mask = np.array([[1]])
    length = len(decoder_attention_mask[0]) + 1
    iter = 0
    new_code = None

    while True:

        logits = decoder_.run(None, {"input_ids": encoded_question_input,
                                     "encoder_attention_mask": encoder_attention_mask,
                                     "encoder_hidden_states": ehs[0],
                                     "decoder_attention_mask": decoder_attention_mask}
                              )

        next_token = logits[0][0][-1]
        new_code = max(enumerate(next_token), key=lambda x: x[1])[0]

        temp_attention = np.array(list(decoder_attention_mask[0]) + [1])
        decoder_attention_mask = temp_attention.astype(np.int64).reshape((batch_size,length))

        temp_input_ids = np.array(list(encoded_question_input[0]) + [new_code])
        encoded_question_input = temp_input_ids.astype(np.int64).reshape((batch_size,length))

        #print(encoded_question_input)

        if new_code == 1:
            break
        else:
            length += 1
            iter += 1

    return encoded_question_input



def get_final_output(tokens,piece_model_path):
    """
    Decodes the token from the decoder.
    Returns Question, Answer and Question + Answer Strings
    """
    s = spm.SentencePieceProcessor(model_file=piece_model_path)
    tokens_list = tokens[0].tolist()
    seperator_idx = tokens_list.index(0)
    question_token = tokens_list[:seperator_idx]
    answer_token = tokens_list[seperator_idx+1:]

    question_string = s.decode(question_token)
    answer_string = s.decode(answer_token)
    combined_string = question_string + " ---> " + answer_string

    return question_string,answer_string,combined_string


def generate_cache_key(*args):
    """
    Generate a unique cache key based on input arrays.
    Combines array data and shapes into a unique hash.
    """
    hasher = hashlib.sha256()
    for arg in args:
        if isinstance(arg, np.ndarray):
            hasher.update(arg.tobytes())
            hasher.update(str(arg.shape).encode())
        else:
            hasher.update(str(arg).encode())
    return hasher.hexdigest()

def run_encoder_with_cache(encoder_, flattened_patches, encoder_attention_mask):
    """
    Runs the encoder with caching support.
    """
    cache_key = generate_cache_key(flattened_patches, encoder_attention_mask)
    if cache_key in encoder_cache:
        return encoder_cache[cache_key]

    output = encoder_.run(None, {
        "flattened_patches": flattened_patches,
        "attention_mask": encoder_attention_mask
    })

    # Cache the result
    encoder_cache[cache_key] = output
    return output

def run_decoder_with_cache(decoder_, encoder_attention_mask, ehs):
    """
    Runs the decoder with caching support.
    """
    encoded_question_input = np.array([[0]])
    decoder_attention_mask = np.array([[1]])
    iter = 0
    length = 1  # Start with length 1 for the initial input sequence

    while True:
        cache_key = generate_cache_key(
            encoded_question_input, decoder_attention_mask, encoder_attention_mask, ehs[0]
        )
        if cache_key in decoder_cache:
            logits = decoder_cache[cache_key]
        else:
            logits = decoder_.run(None, {
                "input_ids": encoded_question_input,
                "encoder_attention_mask": encoder_attention_mask,
                "encoder_hidden_states": ehs[0],
                "decoder_attention_mask": decoder_attention_mask
            })
            decoder_cache[cache_key] = logits

        # Decode the logits
        next_token = logits[0][0][-1]
        new_code = max(enumerate(next_token), key=lambda x: x[1])[0]

        # Update decoder_attention_mask and encoded_question_input
        temp_attention = np.array(list(decoder_attention_mask[0]) + [1])
        decoder_attention_mask = temp_attention.astype(np.int64).reshape((batch_size, length + 1))

        temp_input_ids = np.array(list(encoded_question_input[0]) + [new_code])
        encoded_question_input = temp_input_ids.astype(np.int64).reshape((batch_size, length + 1))

        length += 1  # Increment the sequence length

        if new_code == 1:  # End of sequence token
            break

    return encoded_question_input


def run(paths,question,weightsType=32,cache=False,log=True,cuda=False):
    """
    Wrapper Method
    1. Define Paths and Variables.
    2. Load Encoder and Decoder Models.
    3. Preprocess the Image.
    4. Run the Encoder.
    5. Generate the Decoder Inputs.
    6. Run the Decoder.
    7. Decode the tokens from decoder.
    """
    encoder_model_path = paths["encoderPath"]
    decoder_model_path = paths["decoderPath"]
    piece_model_path = paths["pieceModelPath"]

    font_path = paths["fontPath"]

    #load models
    encoder_,decoder_ = load_models(encoder_model_path,decoder_model_path,cuda)

    #start image processing
    start = time.perf_counter()
    flattened_patches, encoder_attention_mask = preprocess_image(paths["imagePath"], header_text=question, font_path=font_path,weights = weightsType)
    preprocess_image_time = round(time.perf_counter() - start, 3)

    if log:
        print(f"Flattened Patches Shape --> {flattened_patches.shape}\n")
        print(f"Encoder Attention Mask Shape --> {encoder_attention_mask.shape}\n")

    #encoder
    start = time.perf_counter()
    if cache:
        ehs = run_encoder_with_cache(encoder_, flattened_patches, encoder_attention_mask)
    else:
        ehs = run_encoder(encoder_, flattened_patches, encoder_attention_mask)
    encoder_time = round(time.perf_counter() - start,3)

    if log:
        print(f"Encoded Hidden State Shape --> {ehs[0].shape}\n")

    #decoder
    start = time.perf_counter()
    if cache:
        encoded_question_input = run_decoder_with_cache(decoder_,encoder_attention_mask,ehs)
    else:
        encoded_question_input = run_decoder(decoder_,encoder_attention_mask,ehs)

    decoder_time = round(time.perf_counter() - start,3)

    if log:
        print(f"Encoded Question Output --> {encoded_question_input}")
        print(f"Encoded Question Output Shape --> {encoded_question_input.shape}\n")

    #decode tokens to get output
    _,answer,combined = get_final_output(encoded_question_input,piece_model_path)

    if log:
        print(f"Decoded Question --> {question}")
        print(f"Decoded Answer --> {answer}\n")

    return question,answer,encoder_time,decoder_time,preprocess_image_time

