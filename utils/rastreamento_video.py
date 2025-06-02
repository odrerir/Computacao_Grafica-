import cv2
from utils import filtros
def rastrear_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    modo_filtro = 0
    tracker = None
    rastreamento_ativo = False
    bbox = None
    erosao = 0  # 0 = desativado, 1 = binária, 2 = cinza
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    print("\nFiltros disponíveis:")
    print("[0] Sem filtro")
    print("[1] Cinza")
    print("[2] Negativo")
    print("[3] Binária (Otsu)")
    print("[4] Borda (Canny)")
    print("[5] Suavização (Média)")
    print("[6] Erosão (1 - binária, 2 - cinza)")
    print("[7] Dilatação (1 - binária, 2 - cinza)")
    print("[8] Abertura (1 - binária, 2 - cinza)")
    print("[9] Fechamento (1 - binária, 2 - cinza)")
    print("[x] Rastrear objeto selecionado")
    print("[b] Ativar modo binário para morfologia")
    print("[g] Ativar modo cinza para morfologia")
    print("Pressione ESC para sair")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()

        if modo_filtro == 1:
            cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == 2:
            output = 255 - output

        elif modo_filtro == 3:
            cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            output = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == 4:
            cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cinza, 100, 200)
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == 5:
            output = filtros.media(output)

        elif modo_filtro == 6:
            if erosao == 1:
                cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                erodida = cv2.erode(binaria, kernel, iterations=1)
                output = cv2.cvtColor(erodida, cv2.COLOR_GRAY2BGR)
            elif erosao == 2:
                cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                erodida = cv2.erode(cinza, kernel, iterations=1)
                output = cv2.cvtColor(erodida, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == 7:
            if erosao == 1:
                cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                dilatada = cv2.dilate(binaria, kernel, iterations=1)
                output = cv2.cvtColor(dilatada, cv2.COLOR_GRAY2BGR)
            elif erosao == 2:
                cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                dilatada = cv2.dilate(cinza, kernel, iterations=1)
                output = cv2.cvtColor(dilatada, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == 8:
            if erosao == 1:
                cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                abertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
                output = cv2.cvtColor(abertura, cv2.COLOR_GRAY2BGR)
            elif erosao == 2:
                cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                abertura = cv2.morphologyEx(cinza, cv2.MORPH_OPEN, kernel)
                output = cv2.cvtColor(abertura, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == 9:
            if erosao == 1:
                cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                fechamento = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
                output = cv2.cvtColor(fechamento, cv2.COLOR_GRAY2BGR)
            elif erosao == 2:
                cinza = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                fechamento = cv2.morphologyEx(cinza, cv2.MORPH_CLOSE, kernel)
                output = cv2.cvtColor(fechamento, cv2.COLOR_GRAY2BGR)

        elif modo_filtro == "x":
            if not rastreamento_ativo:
                print("\nSelecione o objeto com o mouse e pressione ENTER (ou 'c' para cancelar)...")
                bbox = cv2.selectROI("Selecione o objeto", frame, fromCenter=False, showCrosshair=True)
                x, y, w, h = bbox

                if w == 0 or h == 0:
                    print("Seleção cancelada.")
                    cv2.destroyWindow("Selecione o objeto")
                    modo_filtro = 0
                    print("[0] Sem filtro")
                    print("[1] Cinza")
                    print("[2] Negativo")
                    print("[3] Binária (Otsu)")
                    print("[4] Borda (Canny)")
                    print("[5] Suavização (Média)")
                    print("[6] Erosão (1 - binária, 2 - cinza)")
                    print("[7] Dilatação (1 - binária, 2 - cinza)")
                    print("[8] Abertura (1 - binária, 2 - cinza)")
                    print("[9] Fechamento (1 - binária, 2 - cinza)")
                    print("[x] Rastrear objeto selecionado")
                    print("[b] Ativar modo binário para morfologia")
                    print("[g] Ativar modo cinza para morfologia")
                    print("Pressione ESC para sair")
                else:
                    cv2.destroyWindow("Selecione o objeto")
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    rastreamento_ativo = True
            else:
                sucesso, bbox = tracker.update(frame)
                if sucesso:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(output, "Rastreando", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                else:
                    cv2.putText(output, "Objeto perdido", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    rastreamento_ativo = False
                    tracker = None
                    modo_filtro = 0

        cv2.imshow("Rastreamento", output)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in [ord(str(n)) for n in range(10)]:
            modo_filtro = int(chr(key))
            rastreamento_ativo = False
            tracker = None
        elif key == ord('x'):
            modo_filtro = "x"
        elif key == ord('b'):
            erosao = 1
            print("Modo morfológico: binário")
        elif key == ord('g'):
            erosao = 2
            print("Modo morfológico: cinza")

    cap.release()
    cv2.destroyAllWindows()
