# Como ejecutar 

## Compilando el programa 

### Prerequisitos 

- Compilador de C
- Compilador de C++20
- CMake 3.20 en adelante.
- CUDA
- OpenCl
- glfw.
- OpenGl.

### Compilar 

Para generar una build del proyecto se deben seguir los siguientes pasos: 

En la carpeta raíz ejecutar el siguiente comando: 

```sh
mkdir build # Crear carpeta build
```

Luego 

```sh
cmake -S . -B build # Generar archivos para compilar
```

Por último 

```sh
cmake --build  build # Compilar el proyecto
```

Una vez compilado el proyecto se generará un ejecutable con nombre `GameOfLife`.

### Ejecutando el proyecto 

Una vez compilado el proyecto se debe ejecutar de la siguiente manera: 

```sh
./GameOfLife draw? columns rows hardware/API double_dimension_blocks? block_size
```

Argumentos: 

- `draw`: Un entero, indica si desea tener o una interfaz gráfica para ver el proceso del juego, cualquier valor distinto a 0 activará la interfaz.
- `columns`: Un entero, la cantidad de columnas de la grilla, en otras palabras el valor máximo de la coordenada `X`, cualquier valor menor a 1 generará un error.
- `rows`: Un entero, la cantidad de filas de la grilla, en otras palabras el valor máximo de la coordenada `Y`, cualquier valor menor a 1 generará un error.
- `hardware/API`: Un string, indica que se utilizará para calcular las generaciones de células, valores aceptados: `CUDA`, `OPENCL` o `CPU`.
- `double_dimension_blocks?`: Un entero, indica si se utilizarán bloques de doble dimensión, este argumento solo es utilizado cuando el argumento `hardware/API` no es `CPU`, un valor distinto a 0 utilizará bloques de doble dimensión.
- `block_size`: Un entero, indica el tamaño de los bloques a utilizar en la GPU, este argumento solo es utilizado cuando el argumento `hardware/API` no es `CPU`, un valor menor a 1 generará un error.
