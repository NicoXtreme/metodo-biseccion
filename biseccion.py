import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import sympify, lambdify, symbols

class BiseccionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Método de Bisección Completo")
        self.root.geometry("1000x800")
        
        # Variables para la función y parámetros
        self.funcion = tk.StringVar(value="x**2 - 4")
        self.x_l = tk.DoubleVar(value=0.0)
        self.x_u = tk.DoubleVar(value=3.0)
        self.tolerancia = tk.DoubleVar(value=0.001)
        self.max_iter = tk.IntVar(value=100)
        
        # Configuración de la figura
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(211)
        self.ax_table = self.fig.add_subplot(212)
        self.ax_table.axis('off')
        
        # Canvas para matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Panel de control
        self.crear_panel_control()
        
        # Inicialización
        self.x = symbols('x')
        self.func = None
        self.func_np = None
        self.actualizar_funcion()
    
    def crear_panel_control(self):
        panel = ttk.Frame(self.root, padding="10")
        panel.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Entrada de función
        ttk.Label(panel, text="Función f(x):").grid(row=0, column=0, sticky="e")
        ttk.Entry(panel, textvariable=self.funcion, width=30).grid(row=0, column=1, padx=5, pady=2)
        
        # Parámetros
        ttk.Label(panel, text="Límite inferior (x_l):").grid(row=1, column=0, sticky="e")
        ttk.Entry(panel, textvariable=self.x_l, width=10).grid(row=1, column=1, sticky="w", padx=5)
        
        ttk.Label(panel, text="Límite superior (x_u):").grid(row=2, column=0, sticky="e")
        ttk.Entry(panel, textvariable=self.x_u, width=10).grid(row=2, column=1, sticky="w", padx=5)
        
        ttk.Label(panel, text="Tolerancia:").grid(row=3, column=0, sticky="e")
        ttk.Entry(panel, textvariable=self.tolerancia, width=10).grid(row=3, column=1, sticky="w", padx=5)
        
        ttk.Label(panel, text="Máx iteraciones:").grid(row=4, column=0, sticky="e")
        ttk.Entry(panel, textvariable=self.max_iter, width=10).grid(row=4, column=1, sticky="w", padx=5)
        
        # Botones
        ttk.Button(panel, text="Actualizar Función", command=self.actualizar_funcion).grid(row=0, column=2, padx=5)
        ttk.Button(panel, text="Calcular Raíces", command=self.calcular_raices).grid(row=1, column=2, rowspan=2, padx=5)
        ttk.Button(panel, text="Limpiar", command=self.limpiar).grid(row=3, column=2, rowspan=2, padx=5)
    
    def actualizar_funcion(self):
        try:
            # Convertir string a función matemática
            self.func = sympify(self.funcion.get())
            self.func_np = lambdify(self.x, self.func, modules=['numpy'])
            self.graficar_funcion()
        except Exception as e:
            messagebox.showerror("Error", f"Función inválida: {str(e)}")
    
    def graficar_funcion(self):
        self.ax.clear()
        
        x_l = self.x_l.get()
        x_u = self.x_u.get()
        
        # Generar puntos para la gráfica
        x_vals = np.linspace(min(x_l, x_u)-1, max(x_l, x_u)+1, 400)
        try:
            y_vals = self.func_np(x_vals)
            self.ax.plot(x_vals, y_vals, label=f'f(x) = {str(self.func)}')
            self.ax.axhline(0, color='black', linewidth=0.5)
            self.ax.axvline(0, color='black', linewidth=0.5)
            
            # Marcar límites iniciales
            self.ax.plot(x_l, self.func_np(x_l), 'ro', label='x_l inicial')
            self.ax.plot(x_u, self.func_np(x_u), 'go', label='x_u inicial')
            
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('f(x)')
            self.ax.set_title('Gráfica de la función')
            self.ax.legend()
            self.ax.grid(True)
            
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"No se puede graficar: {str(e)}")
    
    def calcular_raices(self):
        try:
            x_l = self.x_l.get()
            x_u = self.x_u.get()
            tol = self.tolerancia.get()
            max_iter = self.max_iter.get()
            
            if x_l >= x_u:
                raise ValueError("x_l debe ser menor que x_u")
            
            f_l = self.func_np(x_l)
            f_u = self.func_np(x_u)
            
            if f_l * f_u > 0:
                raise ValueError("La función debe tener signos opuestos en x_l y x_u")
            
            # Preparar tabla
            self.ax_table.clear()
            self.ax_table.axis('off')
            
            # Proceso de bisección
            tabla_datos = []
            raiz_aproximada = None
            
            for n in range(max_iter):
                x_r = (x_l + x_u) / 2
                f_r = self.func_np(x_r)
                
                # Agregar datos a la tabla
                tabla_datos.append([
                    n+1,
                    f"{x_l:.6f}",
                    f"{x_u:.6f}",
                    f"{f_l:.6f}",
                    f"{f_u:.6f}",
                    f"{x_r:.6f}",
                    f"{f_r:.6f}"
                ])
                
                # Verificar convergencia
                if abs(f_r) < tol:
                    raiz_aproximada = x_r
                    break
                
                # Actualizar límites
                if f_l * f_r < 0:
                    x_u = x_r
                    f_u = f_r
                else:
                    x_l = x_r
                    f_l = f_r
            else:
                messagebox.showwarning("Advertencia", "Máximo de iteraciones alcanzado")
            
            # Mostrar tabla
            tabla = self.ax_table.table(
                cellText=tabla_datos,
                colLabels=["n", "x_l", "x_u", "f(x_l)", "f(x_u)", "x_r", "f(x_r)"],
                colWidths=[0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
                loc='center',
                cellLoc='center'
            )
            tabla.auto_set_font_size(False)
            tabla.set_fontsize(8)
            tabla.scale(1, 1.2)
            
            # Mostrar raíz en gráfico
            if raiz_aproximada is not None:
                self.ax.plot(raiz_aproximada, 0, 'b*', markersize=10, 
                            label=f'Raíz aprox: {raiz_aproximada:.8f}')
                self.ax.legend()
                messagebox.showinfo("Resultado", 
                                  f"Raíz encontrada: {raiz_aproximada:.8f}\nIteraciones: {n+1}")
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def limpiar(self):
        self.ax.clear()
        self.ax_table.clear()
        self.ax_table.axis('off')
        self.canvas.draw()
        self.graficar_funcion()

if __name__ == "__main__":
    root = tk.Tk()
    app = BiseccionApp(root)
    root.mainloop()
