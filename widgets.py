from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, RoundedRectangle  # Removed SweepGradient
from kivy.utils import get_color_from_hex
from kivy.animation import Animation
from kivy.effects.dampedscroll import DampedScrollEffect
from kivy.uix.behaviors import ButtonBehavior
from kivy.core.window import Window
from kivy.properties import NumericProperty  # Add this import
from kivy.uix.boxlayout import BoxLayout

ORANGE = '#FF9800'
DARK_ORANGE = '#F57C00'
BLACK = '#121212'
WHITE = '#FFFFFF'

class ModernButton(Button):
    scale = NumericProperty(1.0)  # Add scale property
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.font_size = '18sp'
        self.bold = True
        self.color = get_color_from_hex(WHITE)  # Changed to white text
        self.padding = ['20dp', '20dp']
        self.bind(pos=self.update_canvas, size=self.update_canvas, scale=self.update_canvas)
        
        # Add ripple effect
        self.ripple_duration = 0.5
        self.ripple_scale = 2.0
        self.bind(on_press=self.on_press_anim)
        
        # Add shadow and outline
        with self.canvas.before:
            # White outline - increased thickness from 1 to 2 pixels
            Color(1, 1, 1, 0.8)  # White with 80% opacity
            self.outline = RoundedRectangle(
                pos=(self.pos[0] - 2, self.pos[1] - 2),  # Changed from -1 to -2
                size=(self.size[0] + 4, self.size[1] + 4),  # Changed from +2 to +4
                radius=[20,]
            )
            # Shadow
            Color(0, 0, 0, 0.2)
            self.shadow = RoundedRectangle(
                pos=(self.pos[0] + 2, self.pos[1] - 2),
                size=self.size,
                radius=[20,]
            )
            # Original button color and shape
            self.canvas_color = Color(*get_color_from_hex(ORANGE))
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[20,])

    def on_press_anim(self, *args):
        # Scale down button slightly
        anim = (
            Animation(scale=0.95, duration=0.1) + 
            Animation(scale=1.0, duration=0.1)
        )
        anim.start(self)

    def on_hover(self):
        if not self.disabled:
            hover_color = [c * 1.1 for c in get_color_from_hex(ORANGE)]
            self.canvas_color.rgba = hover_color

    def on_leave(self):
        if not self.disabled:
            self.canvas_color.rgba = get_color_from_hex(ORANGE)

    def update_canvas(self, *args):
        # Calculate scaled size and position
        scaled_width = self.width * self.scale
        scaled_height = self.height * self.scale
        
        # Center the scaled button
        x_offset = (self.width - scaled_width) / 2
        y_offset = (self.height - scaled_height) / 2
        
        # Update outline with thicker border
        self.outline.pos = (self.pos[0] + x_offset - 2, self.pos[1] + y_offset - 2)  # Changed from -1 to -2
        self.outline.size = (scaled_width + 4, scaled_height + 4)  # Changed from +2 to +4
        
        self.shadow.pos = (self.pos[0] + x_offset + 2, self.pos[1] + y_offset - 2)
        self.shadow.size = (scaled_width, scaled_height)
        
        self.rect.pos = (self.pos[0] + x_offset, self.pos[1] + y_offset)
        self.rect.size = (scaled_width, scaled_height)

    def on_state(self, widget, value):
        self.canvas_color.rgba = get_color_from_hex(DARK_ORANGE) if value == 'down' else get_color_from_hex(ORANGE)

class CustomLabel(BoxLayout):  # Change from Label to BoxLayout
    def __init__(self, **kwargs):
        # Extract label properties before calling super
        self.label_text = kwargs.pop('text', '')
        self.label_color = kwargs.pop('color', get_color_from_hex(ORANGE))
        self.label_font_size = kwargs.pop('font_size', '16sp')
        self.label_bold = kwargs.pop('bold', False)
        self.label_halign = kwargs.pop('halign', 'left')
        self.label_valign = kwargs.pop('valign', 'middle')
        self.label_markup = kwargs.pop('markup', False)
        
        # Initialize BoxLayout
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = ['5dp', '5dp']
        
        # Add background
        with self.canvas.before:
            Color(*get_color_from_hex(BLACK))
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[15,])
        
        # Create main label with extracted properties
        self.label = Label(
            text=self.label_text,
            color=self.label_color,
            font_size=self.label_font_size,
            bold=self.label_bold,
            halign=self.label_halign,
            valign=self.label_valign,
            markup=self.label_markup,
            size_hint_y=1
        )
        self.add_widget(self.label)

    def update_canvas(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    @property
    def text(self):
        return self.label.text if hasattr(self, 'label') else ''
    
    @text.setter
    def text(self, value):
        if hasattr(self, 'label'):
            self.label.text = value

    def clear_widgets(self, children=None):
        super().clear_widgets(children)
        if children is None and hasattr(self, 'label'):
            self.add_widget(self.label)

    def on_width(self, *args):
        self.label.text_size = (self.width, None)

    def on_text_change(self, *args):
        anim = Animation(opacity=0, duration=0.1) + Animation(opacity=1, duration=0.1)
        anim.start(self)
