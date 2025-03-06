from kivy.uix.widget import Widget
from kivy.properties import NumericProperty
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.utils import get_color_from_hex
from kivy.graphics import Color, Line, Rotate, PushMatrix, PopMatrix

class LoadingSpinner(Widget):
    angle = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (40, 40)
        self.pos_hint = {'center_x': .5, 'center_y': .5}
        
        with self.canvas:
            PushMatrix()
            self.rotation = Rotate(angle=self.angle, origin=self.center)
            Color(*get_color_from_hex('#FF9800'))
            Line(circle=(self.center_x, self.center_y, 15, 0, 360), width=2)
            PopMatrix()
        
        Clock.schedule_once(self.start_spinning, 0)
    # end 
    def start_spinning(self, *args):
        anim = Animation(angle=360, duration=1)
        anim.repeat = True
        anim.start(self)
