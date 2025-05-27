from manim import *

class NeuronWithImageScene(Scene):
    def construct(self):
        self.camera.background_color = WHITE  # Set white background
        self.add_sound("ANN_basic_properties_intro.wav")
        # Load the image and scale/position
        neuron_image = ImageMobject("neuron_c2.png").scale(0.95)
        neuron_image.move_to(RIGHT * 0.1 + UP * 0.3)

        # Get coordinates for positioning
        center = neuron_image.get_center()
        left_point = neuron_image.get_left()
        right_point = neuron_image.get_right()
        top_point = neuron_image.get_top()

        # === INPUT Arrow and Labels ===
        input_arrow = Arrow(start=left_point + RIGHT * 0.5 + DOWN * 1.75,
                            end=left_point+2.5*RIGHT + DOWN * 1.75, color=BLACK)
        input_label = (MathTex("input")
                       .set_color(BLACK)
                       .next_to(input_arrow, DOWN)
                       .shift(LEFT * 0.2))
        weight_label = MathTex("\\times  weight").set_color(BLACK).next_to(input_label, RIGHT)

        # === BIAS Label ===
        bias_label = (MathTex("+  bias").set_color(BLACK)
                      .next_to(weight_label, DOWN).shift(LEFT * 0.35))

        # === OUTPUT Arrow and Label ===
        output_arrow = Arrow(start=right_point - 3*RIGHT + DOWN * 1.75,
                             end=right_point - RIGHT * 1.0 + DOWN * 1.75,
                             color=BLACK)
        output_label = MathTex("= output").set_color(BLACK).next_to(output_arrow, DOWN)

        # Title
        title = Text("Single Artificial Neuron", font_size=30).to_edge(UP)

        # Draw the scene
        self.play(Write(title))
        self.play(FadeIn(neuron_image), run_time=2)
        self.wait(2) # 4 seconds
        self.play(GrowArrow(input_arrow), Write(input_label), run_time=4)
        self.wait(16) # 24
        self.play(Write(weight_label), run_time=3)
        self.play(Write(bias_label), run_time=2)
        self.play(GrowArrow(output_arrow), Write(output_label), run_time=2)
        self.wait(2)