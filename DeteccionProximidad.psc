Algoritmo DeteccionProximidad
	// ============================
	// INICIO
	// ============================
	Escribir 'INICIO: Importar librerías de visión, sonido y web'
	Escribir '-Importar modelo de visión por computadora IA (YOLOV8)'
	// ============================
	// Definir variables
	// ============================
	Escribir "============================"
	Escribir 'Variables definidas: alturas reales, focal cámara'
	Escribir "-altura_persona <- 1.65(UN PROMEDIO)"
	Escribir "-altura_carro <- 1.80"
	Escribir "-focal_pixeles <- 333.33 (FOCAL DE UNA CÁMARA DE CELULAR)"
	
	// ============================
	// DEFINIR OPERACIONES
	// ============================
	Escribir "============================"
	Escribir 'Operaciones disponibles:'
	Escribir '- Detectar cámara'
	Escribir '- Calcular tamaño pixeles del objeto vs pixeles de la imagen total'
	Escribir '- Estimar distancia'
	Escribir '- Pitidos, contadores y cajas de colores  progresivas"
	Escribir "============================"
	
	// ============================
	//   INPUT: Variables de imagen recibidas y DECISIÓN 1: Confianza
	// ============================
	
	Escribir "YOLO Ingresa confianza del objeto (alta)>70% / (baja)<70%: '
	Definir i Como Entero
    Definir confianza Como Real
    Definir clase Como Cadena
    Definir distancia Como Real
    Definir count_personas Como Entero
	
    Escribir "=== INICIO DEL PROCESO ==="
    Escribir "La máquina recibe un frame y empieza a analizar."
	
    count_personas <- 0
	
    Para i <- 1 Hasta 3 Con Paso 1 Hacer
        Escribir "---- Iteración ", i, " del FOR ----"
        Escribir "El FOR garantiza que cada objeto se analice uno por uno."
		
        Escribir "Ingresa confianza (ej: 0.7): "
        Leer confianza
        Escribir "Ingresa clase (persona/vehiculo): "
        Leer clase
        Escribir "Ingresa distancia estimada en metros: "
        Leer distancia
		Escribir "----------------------------------------"
        Escribir "IF: ¿La confianza es suficiente (>= 0.70)?"
        Si confianza < 0.70 Entonces
            Escribir "Decisión: Confianza baja (<0.70). Se DESCARTA la detección."
        Sino
            Escribir "Decisión: Confianza suficiente (>=0.70). Se ANALIZA el objeto."
			
            Escribir "IF: ¿La clase es persona?"
            Si clase = "persona" Entonces
                count_personas <- count_personas + 1
                Escribir "Decisión: Es una PERSONA. Contador de personas = ", count_personas
				
                Escribir "IF/ELSE: Evaluación de distancia para PERSONA."
                Si distancia < 2 Entonces
                    Escribir "Resultado: Persona MUY CERCA (<2m). ALERTA Y CAJA ROJA."
                Sino
                    Si distancia < 5 Entonces
                        Escribir "Resultado: Persona a distancia MEDIA (<5m). ALERTA Y CAJA NARANJA."
                    Sino
                        Si distancia < 10 Entonces
                            Escribir "Resultado: Persona LEJANA (<10m). ALERTA Y CAJA AMARILLA."
                        Sino
                            Escribir "Resultado: Persona SEGURA (>=10m). VERDE."
                        FinSi
                    FinSi
                FinSi
				
            Sino
                Escribir "Decisión: No es persona, se asume VEHÍCULO."
				
                Escribir "IF/ELSE: Evaluación de distancia para VEHÍCULO."
                Si distancia < 3 Entonces
                    Escribir "Resultado: Vehículo MUY CERCA (<3m). ALERTA Y CAJAROJA."
                Sino
                    Si distancia < 6 Entonces
                        Escribir "Resultado: Vehículo a distancia MEDIA (<6m). ALERTA Y CAJA NARANJA."
                    Sino
                        Si distancia < 10 Entonces
                            Escribir "Resultado: Vehículo LEJANO (<10m). ALERTA y CAJA AMARILLA."
                        Sino
                            Escribir "Resultado: Vehículo SEGURO (>=10m). CAJA VERDE."
                        FinSi
                    FinSi
                FinSi
            FinSi
        FinSi
		
    FinPara
	
    Escribir "=== RESULTADOS ==="
    Escribir "Personas detectadas, y respectivo audio diciendo el número: ", count_personas
    Escribir "Cada IF/ELSE mostró cómo la máquina decide según confianza, clase y distancia."
    Escribir "=== FIN DEL PROCESO ==="

FinAlgoritmo

