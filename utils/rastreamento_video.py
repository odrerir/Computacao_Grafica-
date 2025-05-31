import cv2
from utils import filtros  # Substitua por seus próprios filtros se necessário

def rastrear_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    print("\nFiltros disponíveis durante o rastreamento:")
    print("[0] Sem filtro")
    print("[1] Cinza")
    print("[2] Negativo")
    print("[3] Binária (Otsu)")
    print("[4] Borda (Canny)")
    print("[5] Suavização (Média)")
    print("[6] Rastrear objeto selecionado")
    print("Pressione ESC para sair")

    modo_filtro = 0
    tracker = None
    rastreamento_ativo = False
    bbox = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()

        if modo_filtro == 1:  # Cinza
            cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == 2:  # Negativo
            output = 255 - output

        elif modo_filtro == 3:  # Binária (Otsu)
            cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            output = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == 4:  # Borda (Canny)
            cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            edges = filtros.canny(cinza)
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == 5:  # Suavização (Média)
            output = filtros.media(output)

        elif modo_filtro == 6:
            # Se o rastreamento ainda não foi ativado, faça a seleção
            if not rastreamento_ativo:
                print("\nSelecione o objeto com o mouse e pressione ENTER (ou 'c' para cancelar)...")
                bbox = cv2.selectROI("Selecione o objeto", frame, fromCenter=False, showCrosshair=True)
                x, y, w, h = bbox

                if w == 0 or h == 0:
                    print("Seleção cancelada.")
                    cv2.destroyWindow("Selecione o objeto")
                    modo_filtro = 0  # volta para modo sem filtro
                    print("\nFiltros disponíveis durante o rastreamento:")
                    print("[0] Sem filtro")
                    print("[1] Cinza")
                    print("[2] Negativo")
                    print("[3] Binária (Otsu)")
                    print("[4] Borda (Canny)")
                    print("[5] Suavização (Média)")
                    print("[6] Rastrear objeto selecionado")
                    print("Pressione ESC para sair")
                else:
                    cv2.destroyWindow("Selecione o objeto")
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    rastreamento_ativo = True
            else:
                # Atualiza rastreamento a cada frame
                sucesso, bbox = tracker.update(frame)
                if sucesso:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(output, "Rastreando", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                else:
                    cv2.putText(output, "Objeto perdido", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    rastreamento_ativo = False
                    tracker = None
                    modo_filtro = 0  # volta para modo sem filtro

        cv2.imshow("Rastreamento", output)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC para sair
            break
        elif key in map(ord, "0123456"):
            modo_filtro = int(chr(key))
            rastreamento_ativo = False
            tracker = None  # resetar rastreador ao trocar filtro

    cap.release()
    cv2.destroyAllWindows()
