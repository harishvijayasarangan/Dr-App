import onnxruntime
import numpy as  np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button  # Add this import
from kivy.graphics import Color, Rectangle  # Add this import
from kivy.uix.popup import Popup
from kivy.uix.camera import Camera
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserListView
from kivy.utils import get_color_from_hex
from kivy.uix.image import Image
from kivy.clock import Clock
from PIL import Image as PILImage
from widgets import ModernButton, CustomLabel
import io
import os
import gc
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.image import Image as KivyImage
from loading import LoadingSpinner
from kivy.animation import Animation
from functools import partial
from kivy.uix.progressbar import ProgressBar
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ListProperty
from kivy.uix.label import Label  # Add this import
from kivy.uix.floatlayout import FloatLayout  # Add this import
from kivy.uix.scrollview import ScrollView  # Add this import

labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 2
session = onnxruntime.InferenceSession('dr-model.onnx', session_options)

def transform_image(image_path):
    with PILImage.open(image_path) as img:
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        img_array = np.transpose(img_array, (2, 0, 1))
        return np.expand_dims(img_array, axis=0).astype(np.float32)

class CustomProgressBar(Widget):
    value = NumericProperty(0.0)  
    max = NumericProperty(100.0)  
    background_color = ListProperty([0.1, 0.1, 0.1, 1])  # Dark background
    color = ListProperty([1, 0.6, 0, 1])  # Orange color

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self.draw, size=self.draw, value=self.draw)

    def draw(self, *args):
        self.canvas.clear()
        with self.canvas:
            # Draw background
            Color(*self.background_color)
            Rectangle(pos=self.pos, size=self.size)
            
            # Draw progress
            Color(*self.color)
            width = self.width * (self.value / self.max)
            Rectangle(pos=self.pos, size=(width, self.height))

class BackgroundScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=0, spacing=0)  # Remove padding
        
        # Background image with full coverage
        bg_image = KivyImage(
            source='b.png',
            allow_stretch=True,
            keep_ratio=False,
            size_hint=(1, 1),  # Full size
            pos_hint={'center_x': 0.5, 'center_y': 0.5}  # Center position
        )
        
        # Create a BoxLayout for buttons at the bottom
        button_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.15,
            spacing=20,
            padding=[20, 0, 20, 20]
        )
        
        # Create buttons
        gallery_btn = ModernButton(
            text='Upload Image',
            background_color=get_color_from_hex('#FF9800')
        )
        camera_btn = ModernButton(
            text='Take Photo',
            background_color=get_color_from_hex('#FF9800')
        )
        
        # Add buttons to button layout
        button_layout.add_widget(gallery_btn)
        button_layout.add_widget(camera_btn)
        
        # Create a FloatLayout to overlay buttons on the image
        from kivy.uix.floatlayout import FloatLayout
        float_layout = FloatLayout(size_hint=(1, 1))  # Full size
        
        # Add image to float layout
        float_layout.add_widget(bg_image)
        
        # Add button layout to float layout at the bottom
        button_layout.pos_hint = {'center_x': 0.5, 'y': 0.05}
        button_layout.size_hint = (0.9, 0.1)
        float_layout.add_widget(button_layout)
        
        
        layout.add_widget(float_layout)
        self.add_widget(layout)
        
        # Bind buttons to app methods
        gallery_btn.bind(on_press=self.show_file_chooser)
        camera_btn.bind(on_press=self.show_camera)
    
    def show_file_chooser(self, instance):
        app = App.get_running_app()
        app.show_file_chooser(instance)
    
    def show_camera(self, instance):
        app = App.get_running_app()
        app.show_camera(instance)

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=0, spacing=20)  # Remove padding

ORANGE = '#FF9800'
BLACK = '#121212'
WHITE = '#FFFFFF'

class DRDetectionApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = None
        self.camera_popup = None
        self.screen_manager = ScreenManager(transition=FadeTransition(duration=0.5))
        
    def build(self):
        try:
            Window.size = (400, 700)
            Window.clearcolor = get_color_from_hex(BLACK)  # Black background
            
            # Create screens
            background_screen = BackgroundScreen(name='background')
            main_screen = MainScreen(name='main')
            
            # Add screens to manager
            self.screen_manager.add_widget(background_screen)
            self.screen_manager.add_widget(main_screen)
            
            # Setup main screen widgets
            self._setup_widgets(main_screen.layout)
            main_screen.add_widget(main_screen.layout)
            
            return self.screen_manager
        except Exception as e:
            print(f"Build error: {str(e)}")
            return None

    def _setup_widgets(self, layout):
        self.layout = layout
        # Zero spacing between elements
        self.layout.padding = [0, 0, 0, 0]
        self.layout.spacing = 0
        
        # Tiny header
        self.layout.add_widget(CustomLabel(
            text='DR Detection',
            font_size='20sp',
            bold=True,
            size_hint_y=0.05,  # Small header
            color=get_color_from_hex(ORANGE),
            padding=[0, 0]
        ))
        
        # Large image container
        self.img_container = BoxLayout(
            size_hint_y=0.45,  # Large image (45% of screen)
            padding=0,
            spacing=0
        )
        with self.img_container.canvas.before:
            Color(0, 0, 0, 1)
            self.img_bg = Rectangle(pos=self.img_container.pos, size=self.img_container.size)
        
        self.img_container.bind(pos=self._update_rect, size=self._update_rect)
        
        self.img_display = Image(
            size_hint=(1, 1),  # Take full container space
            allow_stretch=True,
            keep_ratio=True,
            fit_mode="contain"
        )
        self.img_container.add_widget(self.img_display)
        self.layout.add_widget(self.img_container)
        
        # Results right below image
        self.result_label = CustomLabel(
            size_hint_y=0.90,  # Results take 45% too
            size_hint_x=1,  # Full width
            padding=[5, 0]  # Minimal padding
        )
        self.layout.add_widget(self.result_label)
        
        # Tiny bottom space for buttons
        button_layout = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.05,  # Minimal height
            spacing=10,
            padding=[10, 0, 10, 5]
        )
        # ...rest of button layout code...

    # Add new method for updating rectangle
    def _update_rect(self, instance, value):
        self.img_bg.pos = instance.pos
        self.img_bg.size = instance.size

    def show_file_chooser(self, instance):
        # Remove plyer and use native file chooser directly
        content = BoxLayout(orientation='vertical')
        file_chooser = FileChooserListView(
            path=os.path.expanduser('~'),
            filters=['*.png', '*.jpg', '*.jpeg']
        )
        content.add_widget(file_chooser)
        
        select_button = ModernButton(  # Use ModernButton instead of Button
            text='Select',
            size_hint_y=0.2
        )
        content.add_widget(select_button)
        
        popup = Popup(
            title='Choose Image',
            content=content,
            size_hint=(0.9, 0.9)
        )
        
        def select(instance):
            if file_chooser.selection:
                self.process_image(file_chooser.selection[0])
                popup.dismiss()
        
        select_button.bind(on_press=select)
        popup.open()

    def _handle_selection(self, selection):
        if selection and len(selection) > 0:
            self.process_image(selection[0])
    
    def show_camera(self, instance):
        try:
            content = BoxLayout(orientation='vertical')
            self.camera = Camera(play=True, resolution=(640, 480))
            content.add_widget(self.camera)
            
            button_layout = BoxLayout(size_hint_y=0.2, spacing=10)
            capture_button = ModernButton(text='Capture')
            cancel_button = ModernButton(text='Cancel')
            button_layout.add_widget(capture_button)
            button_layout.add_widget(cancel_button)
            content.add_widget(button_layout)
            
            self.camera_popup = Popup(
                title='Take Photo',
                content=content,
                size_hint=(0.9, 0.9),
                auto_dismiss=False  # Prevent accidental dismissal
            )
            
            def cleanup_camera(*args):
                try:
                    if self.camera:
                        self.camera.play = False
                    if self.camera_popup:
                        self.camera_popup.dismiss()
                    Clock.schedule_once(lambda dt: self._final_cleanup(), 0.1)
                except Exception as e:
                    print(f"Cleanup error: {str(e)}")

            def _capture(instance):
                if self.camera:
                    try:
                        self.camera.export_to_png('temp_capture.png')
                        Clock.schedule_once(lambda dt: self.process_image('temp_capture.png'), 0.1)
                        cleanup_camera()
                    except Exception as e:
                        print(f"Capture error: {str(e)}")
            
            capture_button.bind(on_press=_capture)
            cancel_button.bind(on_press=cleanup_camera)
            
            self.camera_popup.open()
        except Exception as e:
            print(f"Camera error: {str(e)}")

    def _final_cleanup(self, *args):
        try:
            self.camera = None
            gc.collect()
        except Exception as e:
            print(f"Final cleanup error: {str(e)}")

    def process_image(self, image_path):
        try:
            # Show loading spinner
            spinner = LoadingSpinner()
            self.layout.add_widget(spinner)
            
            # Schedule actual processing
            Clock.schedule_once(partial(self._do_process_image, image_path, spinner), 0.1)
            
        except Exception as e:
            self.result_label.text = f"[color={WHITE}]Error: {str(e)}[/color]"
            self.result_label.markup = True

    def _do_process_image(self, image_path, spinner, *args):
        try:
            # Switch to main screen with fade
            self.screen_manager.transition.duration = 0.3
            self.screen_manager.current = 'main'
            
            # Then process the image
            self.img_display.source = image_path

            input_tensor = transform_image(image_path)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            prediction = session.run([output_name], {input_name: input_tensor})[0][0]

            exp_preds = np.exp(prediction - np.max(prediction))
            probabilities = exp_preds / exp_preds.sum()

            # Calculate bar width based on window width
            max_bar_width = 60  # Maximum width for highest probability
            
            # Sort indices by probability for proper scaling
            top_indices = np.argsort(probabilities)[-5:][::-1]
            max_prob = probabilities[top_indices[0]]
            
            result_lines = [
                f"[b][size=32][color={WHITE}]Diagnosis Results[/color][/size][/b]\n\n",  # Increased from 28 to 32
                "[size=28]"  # Increased base size from 24 to 28
            ]

            # Create a vertical layout for each result
            from kivy.uix.boxlayout import BoxLayout
            results_layout = BoxLayout(
                orientation='vertical',
                spacing=2,  # Minimal spacing
                padding=[5, 0, 5, 0],
                size_hint=(1, 1)  # Take full space
            )
            
            # Title right at top
            title = CustomLabel(
                text=f"[b][size=36][color={WHITE}]Diagnosis Results[/color][/size][/b]",
                markup=True,
                size_hint_y=0.15,
                padding=[0, 0]
            )
            results_layout.add_widget(title)

            # Process all classes in order (0 to 4)
            prob_rows = []
            for i in range(5):  # Always show all 5 classes
                prob = probabilities[i]
                
                # Create row layout
                row = BoxLayout(
                    orientation='horizontal',
                    size_hint_y=0.17,
                    spacing=2
                )
                
                # Create widgets for the row
                label = CustomLabel(
                    text=f"[color={WHITE}][b][size=34]{labels[i]}[/size][/b][/color]",
                    size_hint_x=0.4,
                    markup=True
                )
                row.add_widget(label)
                
                prog = CustomProgressBar(
                    size_hint_x=0.35,
                    height='50dp'  # Taller bars
                )
                row.add_widget(prog)
                
                percent = CustomLabel(
                    text=f"[color={WHITE}][b][size=32]{prob:.1%}[/b][/color]",
                    size_hint_x=0.25,
                    markup=True
                )
                row.add_widget(percent)
                
                results_layout.add_widget(row)
                prob_rows.append((prob, row))
            
            # Clear and add results directly without ScrollView
            self.result_label.clear_widgets()
            self.result_label.add_widget(results_layout)
            
            # Animate progress bars
            for prob, row in prob_rows:
                for widget in row.children:
                    if isinstance(widget, CustomProgressBar):
                        widget.value = 0.0
                        prob_value = float(prob * 100)
                        anim = Animation(value=prob_value, duration=0.8, t='out_quad')
                        anim.start(widget)
            
            # Remove spinner
            self.layout.remove_widget(spinner)
            
        except Exception as e:
            print(f"Process error: {str(e)}")
            self.result_label.text = f"[color={WHITE}]Error: {str(e)}[/color]"
            self.result_label.markup = True
            if spinner in self.layout.children:
                self.layout.remove_widget(spinner)

    def get_application_name(self):
        return "drdetection"

    def on_stop(self):
        if self.camera:
            self.camera.play = False
            self.camera = None
        if os.path.exists('temp_capture.png'):
            os.remove('temp_capture.png')
        gc.collect()

if __name__ == '__main__':
    try:
        DRDetectionApp().run()
    except Exception as e:
        print(f"Application error: {str(e)}")