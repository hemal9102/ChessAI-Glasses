import cv2
import chess
import chess.engine
import numpy as np
import os
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

VIDEO_PATH = "test-images/2323.mp4"
STOCKFISH_PATH = r"H:\Minor Project 2\ChessAI-Glasses\Frontend\stockfish\stockfish.exe"
MODEL_PATH = r"chess-model-yolov8m.pt"  # Use your actual model file
CONFIDENCE_THRESHOLD = 0.65

# =============================================================================
# BOARD DETECTION & WARPING
# =============================================================================

def detect_board_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    board_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                board_cnt = approx
                max_area = area
    if board_cnt is not None:
        pts = board_cnt.reshape(4, 2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(diff)]
        ordered[3] = pts[np.argmax(diff)]
        return ordered.tolist()
    return None

def warp_board(frame, corners, size=400):
    src = np.array(corners, dtype="float32")
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (size, size))
    return warped, M

def get_square_pixel(square, M_inv, board_size=8, board_px=400):
    file = ord(square[0]) - ord('a')
    rank = 8 - int(square[1])
    x = int((file + 0.5) * (board_px / board_size))
    y = int((rank + 0.5) * (board_px / board_size))
    pt = np.array([[[x, y]]], dtype='float32')
    mapped = cv2.perspectiveTransform(pt, M_inv)[0][0]
    return int(mapped[0]), int(mapped[1])

# =============================================================================
# PIECE DETECTION
# =============================================================================

def extract_board_and_pieces(warped_board, model):
    results = model.predict(source=warped_board, conf=CONFIDENCE_THRESHOLD, imgsz=400, verbose=False)
    chess_pieces = []
    class_map = model.names
    fen_map = {
        'bishop': 'B', 'black-bishop': 'b', 'black-king': 'k', 'black-knight': 'n',
        'black-pawn': 'p', 'black-queen': 'q', 'black-rook': 'r', 'white-bishop': 'B',
        'white-king': 'K', 'white-knight': 'N', 'white-pawn': 'P', 'white-queen': 'Q', 'white-rook': 'R'
    }
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        col = int(center_x // (400 / 8))
        row = int(center_y // (400 / 8))
        square_name = chr(ord('a') + col) + str(8 - row)
        class_id = int(result.cls.cpu().numpy())
        class_name = class_map[class_id]
        fen_char = fen_map.get(class_name)
        if fen_char:
            chess_pieces.append((square_name, class_name))
        else:
            print(f"Warning: Unmapped class detected: {class_name} (ID: {class_id})")
    return chess_pieces

# =============================================================================
# VALIDATION & STOCKFISH INTEGRATION
# =============================================================================

def validate_chess_position(chess_pieces):
    piece_count = {'white-king': 0, 'black-king': 0, 'white-pawn': 0, 'black-pawn': 0, 'white': 0, 'black': 0}
    for sq, name in chess_pieces:
        if name == 'white-king':
            piece_count['white-king'] += 1
            piece_count['white'] += 1
        elif name == 'black-king':
            piece_count['black-king'] += 1
            piece_count['black'] += 1
        elif name == 'white-pawn':
            piece_count['white-pawn'] += 1
            piece_count['white'] += 1
            if sq[1] in ('1', '8'):
                return False
        elif name == 'black-pawn':
            piece_count['black-pawn'] += 1
            piece_count['black'] += 1
            if sq[1] in ('1', '8'):
                return False
        elif name.startswith('white'):
            piece_count['white'] += 1
        elif name.startswith('black'):
            piece_count['black'] += 1
    if piece_count['white-king'] != 1 or piece_count['black-king'] != 1:
        return False
    if piece_count['white-pawn'] > 8 or piece_count['black-pawn'] > 8:
        return False
    if piece_count['white'] > 16 or piece_count['black'] > 16:
        return False
    return True

def build_board_from_pieces(chess_pieces):
    board = chess.Board(None)
    piece_map = {
        'white-king': chess.Piece.from_symbol('K'),
        'black-king': chess.Piece.from_symbol('k'),
        'white-queen': chess.Piece.from_symbol('Q'),
        'black-queen': chess.Piece.from_symbol('q'),
        'white-rook': chess.Piece.from_symbol('R'),
        'black-rook': chess.Piece.from_symbol('r'),
        'white-bishop': chess.Piece.from_symbol('B'),
        'black-bishop': chess.Piece.from_symbol('b'),
        'white-knight': chess.Piece.from_symbol('N'),
        'black-knight': chess.Piece.from_symbol('n'),
        'white-pawn': chess.Piece.from_symbol('P'),
        'black-pawn': chess.Piece.from_symbol('p'),
    }
    for sq, name in chess_pieces:
        if name in piece_map:
            board.set_piece_at(chess.parse_square(sq), piece_map[name])
    return board

# =============================================================================
# MAIN LOOP
# =============================================================================

if __name__ == '__main__':
    print(f"Loading YOLOv8 model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print(f"Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}. Check the path and file existence.")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {width}x{height} @ {fps} FPS")

    print(f"Opening Stockfish engine from: {STOCKFISH_PATH}")
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Error: Stockfish executable not found at {STOCKFISH_PATH}")
        exit(1)

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            frame_count += 1
            print(f"Processing frame {frame_count}...")

            corners = detect_board_corners(frame)
            if corners is None:
                print("  - No board corners detected in this frame.")
                cv2.imshow("Chess Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            warped, M = warp_board(frame, corners)
            M_inv = cv2.getPerspectiveTransform(
                np.array([[0,0],[399,0],[399,399],[0,399]], dtype="float32"),
                np.array(corners, dtype="float32")
            )

            chess_pieces = extract_board_and_pieces(warped, model)
            print(f"  - Detected {len(chess_pieces)} pieces: {chess_pieces}")

            if not validate_chess_position(chess_pieces):
                print("  - Invalid chess position detected, skipping Stockfish analysis for this frame.")
                cv2.imshow("Chess Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            board = build_board_from_pieces(chess_pieces)
            fen = board.fen()
            print(f"  - Generated FEN: {fen}")

            try:
                result = engine.analyse(board, chess.engine.Limit(time=0.1))
                best_move = result['pv'][0]
                print(f"  - Best move suggested by Stockfish: {best_move}")

                from_sq = chess.square_name(best_move.from_square)
                to_sq = chess.square_name(best_move.to_square)

                start_pt = get_square_pixel(from_sq, M_inv, board_size=8, board_px=400)
                end_pt = get_square_pixel(to_sq, M_inv, board_size=8, board_px=400)

                cv2.arrowedLine(frame, start_pt, end_pt, (0, 255, 0), 4, tipLength=0.3)
                cv2.putText(frame, f"Move: {best_move}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"  - Error during Stockfish analysis: {e}")
                cv2.putText(frame, "Stockfish Error", (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Chess Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User interrupted.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")
