"""Generates the Study Assistant's source corpus: eight short ECE lecture-note PDFs.

The corpus is generated rather than downloaded so the repo is self-contained and every fact a
model might retrieve is checkable by eye. Each document is written to be *answerable*: it contains
specific numbers, formulas and units, not vague prose, because a RAG evaluation is only as good as
the ground truth it can be scored against.

    python data/build_corpus.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

HERE = Path(__file__).resolve().parent

H1 = ParagraphStyle("H1", parent=getSampleStyleSheet()["Heading1"], fontSize=15, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=getSampleStyleSheet()["Heading2"], fontSize=11.5,
                    spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=getSampleStyleSheet()["BodyText"], fontSize=9.5,
                      leading=13, alignment=TA_JUSTIFY, spaceAfter=4)

DOCS: list[tuple[str, str, list[tuple[str, list[str]]]]] = [
    ("01_signals_and_systems", "Signals and Systems - Lecture Notes", [
        ("1. Signals and Classification", [
            "A signal is a function of one or more independent variables that carries information "
            "about the state of a physical system. Continuous-time signals are defined for every "
            "value of t; discrete-time signals are defined only at integer indices n. A signal "
            "x(t) is periodic with period T if x(t + T) = x(t) for all t, and the fundamental "
            "frequency is f0 = 1/T, measured in hertz. The angular frequency is omega0 = 2*pi*f0 "
            "radians per second.",
            "An even signal satisfies x(t) = x(-t). An odd signal satisfies x(t) = -x(-t). Every "
            "signal can be decomposed into an even part (x(t) + x(-t))/2 and an odd part "
            "(x(t) - x(-t))/2. The unit impulse delta(t) has unit area and zero width; the unit "
            "step u(t) is 0 for t < 0 and 1 for t >= 0, and du(t)/dt = delta(t).",
        ]),
        ("2. LTI Systems and Convolution", [
            "A system is linear if it obeys superposition and time-invariant if a time shift in "
            "the input produces the same shift in the output. A Linear Time-Invariant (LTI) "
            "system is completely characterised by its impulse response h(t).",
            "The output is the convolution of the input with the impulse response: "
            "y(t) = x(t) * h(t) = integral of x(tau) h(t - tau) d(tau). In discrete time, "
            "y[n] = sum over k of x[k] h[n - k]. Convolution is commutative and associative, so "
            "systems in cascade can be combined by convolving their impulse responses h(t) = "
            "h1(t) * h2(t).",
            "For an LTI system the output to a complex exponential e^(j*omega*t) is the same "
            "exponential scaled by the frequency response H(omega). This is the eigenfunction "
            "property and it is the reason the Fourier transform is useful.",
        ]),
        ("3. Fourier Analysis", [
            "The continuous-time Fourier transform pair is X(omega) = integral x(t) e^(-j omega t) "
            "dt, and x(t) = (1/2pi) integral X(omega) e^(j omega t) d(omega). Useful pairs: a "
            "rectangular pulse of width T transforms to a sinc function T*sinc(omega*T/2), and an "
            "impulse transforms to a constant.",
            "The Fourier series applies to periodic signals and expresses them as a sum of "
            "harmonics at integer multiples of f0. The Laplace transform generalises the Fourier "
            "transform to complex frequency s = sigma + j omega and is used to analyse systems "
            "with initial conditions and unstable responses.",
        ]),
        ("4. Sampling and the Nyquist Theorem", [
            "The sampling theorem states that a bandlimited signal with maximum frequency f_max "
            "can be reconstructed exactly from its samples if the sampling rate fs is greater than "
            "2*f_max. The rate 2*f_max is the Nyquist rate, and fs/2 is the Nyquist frequency. "
            "If fs is below the Nyquist rate, aliasing occurs and high frequencies appear as "
            "spurious low frequencies.",
            "An anti-aliasing low-pass filter is placed before the sampler to remove components "
            "above fs/2. The reconstruction (interpolation) formula is x(t) = sum over n of "
            "x[nT] sinc((t - nT)/T), where T = 1/fs is the sampling interval.",
        ]),
    ]),
    ("02_digital_electronics", "Digital Electronics - Lecture Notes", [
        ("1. Boolean Algebra and Logic Gates", [
            "The basic gates are AND, OR and NOT. The universal gates are NAND and NOR, each of "
            "which can implement any Boolean function. XOR (exclusive-OR) outputs 1 only when its "
            "two inputs differ; XNOR is its complement. De Morgan's theorems are: "
            "NOT(A AND B) = (NOT A) OR (NOT B) and NOT(A OR B) = (NOT A) AND (NOT B).",
            "A minterm is a product term containing every variable once. Sum-of-products (SOP) "
            "form is a sum of minterms; product-of-sums (POS) is a product of maxterms. A "
            "Karnaugh map is a rectangular grid used to simplify Boolean expressions; adjacent "
            "cells that differ in exactly one variable can be grouped in powers of two (1, 2, 4, "
            "8, ...).",
        ]),
        ("2. Combinational Logic", [
            "A half adder adds two bits and produces Sum = A XOR B and Carry = A AND B. A full "
            "adder adds three bits (A, B and carry-in) and produces Sum and carry-out; it can be "
            "built from two half adders and an OR gate. A multiplexer (MUX) with 2^n data inputs "
            "needs n select lines. A decoder with n inputs has 2^n outputs.",
        ]),
        ("3. Sequential Logic and Flip-Flops", [
            "Sequential circuits have memory. A latch is level-sensitive; a flip-flop is "
            "edge-triggered. The RS latch has an invalid state when both inputs are asserted. The "
            "D flip-flop stores the input on a clock edge: Q(next) = D. The JK flip-flop removes "
            "the invalid state: when J = K = 1 the output toggles. The T flip-flop toggles when "
            "T = 1 and holds when T = 0.",
            "Setup time is the interval before the active clock edge during which the data input "
            "must be stable. Hold time is the interval after the edge during which the input must "
            "remain stable. Violating these causes metastability.",
        ]),
        ("4. Counters and Registers", [
            "A mod-N counter has N distinct states and divides the input frequency by N. An "
            "n-bit binary ripple counter divides by 2^n. A ring counter circulates a single 1 "
            "through a shift register; a Johnson (twisted-ring) counter has 2n states for n "
            "flip-flops. A shift register moves data one position per clock and can convert "
            "between serial and parallel formats.",
        ]),
    ]),
    ("03_control_systems", "Control Systems - Lecture Notes", [
        ("1. System Modelling", [
            "A linear time-invariant system is described by a transfer function "
            "G(s) = Y(s)/U(s) evaluated with zero initial conditions, where s is the Laplace "
            "variable. The order of the system is the degree of the denominator polynomial. The "
            "poles are the roots of the denominator and govern the transient response; the zeros "
            "are the roots of the numerator.",
            "A first-order system is written G(s) = K/(1 + tau*s), where K is the DC gain and tau "
            "is the time constant. Its step response reaches 63.2 percent of the final value at "
            "t = tau, and settles to within 2 percent in about 4*tau.",
        ]),
        ("2. Second-Order Systems", [
            "The standard second-order form is G(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + "
            "omega_n^2), where omega_n is the natural frequency and zeta is the damping ratio. "
            "For zeta < 1 the system is underdamped and the peak overshoot is "
            "Mp = exp(-pi*zeta / sqrt(1 - zeta^2)). For zeta = 1 it is critically damped, and "
            "for zeta > 1 overdamped.",
            "The 2 percent settling time is approximately Ts = 4/(zeta*omega_n). The damped "
            "natural frequency is omega_d = omega_n*sqrt(1 - zeta^2).",
        ]),
        ("3. Stability", [
            "A system is stable if all poles lie in the left half of the s-plane (negative real "
            "part). The Routh-Hurwitz criterion determines stability from the coefficients of the "
            "characteristic polynomial without solving for the roots. The number of sign changes "
            "in the first column of the Routh array equals the number of right-half-plane poles.",
            "A marginally stable or unstable system has poles on the imaginary axis or in the "
            "right half plane respectively.",
        ]),
        ("4. PID Control", [
            "A PID controller produces "
            "u(t) = Kp*e(t) + Ki*integral of e dtau + Kd*de/dt. The proportional term Kp reduces "
            "rise time but leaves steady-state error; the integral term Ki removes steady-state "
            "error but can increase overshoot; the derivative term Kd reduces overshoot and "
            "improves settling time but amplifies noise.",
            "The Ziegler-Nichols method tunes gains from the ultimate gain Ku and ultimate period "
            "Tu: for a PID controller Kp = 0.6*Ku, Ti = 0.5*Tu and Td = 0.125*Tu.",
        ]),
    ]),
    ("04_communication_systems", "Communication Systems - Lecture Notes", [
        ("1. Amplitude Modulation", [
            "In Amplitude Modulation (AM) the carrier amplitude varies with the message: "
            "s(t) = Ac*(1 + m*cos(2*pi*fm*t))*cos(2*pi*fc*t), where m is the modulation index. "
            "For distortion-free demodulation m must satisfy 0 < m <= 1. The bandwidth of a "
            "standard AM signal is twice the message bandwidth, BW = 2*fm.",
            "The modulation index is m = Am/Ac, the ratio of message amplitude to carrier "
            "amplitude. An AM signal contains the carrier and two sidebands; suppressing the "
            "carrier gives double-sideband suppressed-carrier (DSB-SC), which requires coherent "
            "detection.",
        ]),
        ("2. Angle Modulation", [
            "In Frequency Modulation (FM) the instantaneous frequency deviates with the message. "
            "The peak frequency deviation is delta_f, and the modulation index is beta = "
            "delta_f/fm. Carson's rule approximates the bandwidth as BW = 2*(delta_f + fm), or "
            "2*(beta + 1)*fm.",
            "FM is more robust to amplitude noise than AM because the information is carried in "
            "the frequency, but it requires a wider bandwidth.",
        ]),
        ("3. Information Theory", [
            "Channel capacity in the presence of additive white Gaussian noise is given by the "
            "Shannon-Hartley theorem: C = B*log2(1 + S/N) bits per second, where B is the channel "
            "bandwidth in hertz and S/N is the signal-to-noise power ratio (a linear ratio, not "
            "in decibels).",
            "The Shannon limit states that error-free communication is possible only for rates "
            "below C. Doubling the bandwidth does not double capacity, because adding bandwidth "
            "also adds noise power.",
        ]),
        ("4. Digital Modulation", [
            "Common digital schemes include Amplitude Shift Keying (ASK), Frequency Shift Keying "
            "(FSK) and Phase Shift Keying (PSK). Binary PSK (BPSK) uses two phases 180 degrees "
            "apart. Quadrature PSK (QPSK) carries two bits per symbol using four phases. "
            "Quadrature Amplitude Modulation (QAM) combines amplitude and phase; 16-QAM carries "
            "4 bits per symbol and 64-QAM carries 6 bits per symbol.",
        ]),
    ]),
    ("05_semiconductor_devices", "Semiconductor Devices - Lecture Notes", [
        ("1. Semiconductor Fundamentals", [
            "Silicon has four valence electrons. Doping with a pentavalent impurity (phosphorus) "
            "gives n-type material with electrons as majority carriers; a trivalent impurity "
            "(boron) gives p-type material with holes as majority carriers. The intrinsic carrier "
            "concentration of silicon at 300 K is about 1.5e10 per cubic centimetre.",
        ]),
        ("2. The PN Junction Diode", [
            "A forward-biased diode conducts above the barrier potential, about 0.7 volts for "
            "silicon and 0.3 volts for germanium. The diode current is "
            "I = Is*(exp(V/(n*Vt)) - 1), where Vt is the thermal voltage, approximately 26 mV at "
            "room temperature, and n is the ideality factor (close to 1 for silicon).",
            "Reverse bias produces only a small leakage current. A Zener diode operates in reverse "
            "breakdown and is used for voltage regulation; an LED is a forward-biased diode that "
            "emits light.",
        ]),
        ("3. Bipolar Junction Transistor", [
            "A BJT has three terminals: emitter, base and collector. In the active region the "
            "emitter-base junction is forward biased and the collector-base junction is reverse "
            "biased. The current gain beta (h_FE) relates collector current to base current: "
            "Ic = beta*Ib. The relation between alpha and beta is alpha = beta/(beta + 1) and "
            "beta = alpha/(1 - alpha).",
            "In saturation the transistor behaves as a closed switch with Vce about 0.2 V; in "
            "cut-off it behaves as an open switch.",
        ]),
        ("4. Field Effect Transistors", [
            "A MOSFET is a voltage-controlled device. In an n-channel enhancement MOSFET, "
            "conduction begins when the gate-source voltage exceeds the threshold voltage Vth. "
            "In the saturation region the drain current is "
            "Id = 0.5*kn*(Vgs - Vth)^2. The transconductance is gm = 2*Id/(Vgs - Vth).",
            "MOSFETs have very high input impedance (gates draw almost no current), low power "
            "dissipation and are the basis of CMOS logic.",
        ]),
    ]),
    ("06_transmission_lines", "Transmission Lines and Electromagnetics - Lecture Notes", [
        ("1. Transmission Line Basics", [
            "A transmission line is characterised by its characteristic impedance Z0 = "
            "sqrt(L/C), where L and C are the inductance and capacitance per unit length. Coaxial "
            "cable commonly uses Z0 = 50 ohms; twisted pair used for broadcast television uses "
            "75 ohms.",
            "The velocity of propagation is v = 1/sqrt(L*C) and is often expressed as a fraction "
            "of the speed of light c = 3e8 m/s. The wavelength on the line is lambda = v/f.",
        ]),
        ("2. Reflections and Matching", [
            "When a line is terminated in a load ZL that is not equal to Z0, part of the incident "
            "wave reflects. The reflection coefficient is Gamma = (ZL - Z0)/(ZL + Z0). The "
            "Voltage Standing Wave Ratio is VSWR = (1 + |Gamma|)/(1 - |Gamma|); a perfectly "
            "matched line has VSWR = 1.",
            "Maximum power transfer occurs when ZL equals the complex conjugate of the source "
            "impedance. A quarter-wave transformer of impedance Z0' = sqrt(Z0*ZL) matches a load "
            "to a line.",
        ]),
        ("3. Maxwell and Propagation", [
            "Maxwell's equations describe how electric and magnetic fields are generated and "
            "altered. In free space electromagnetic waves travel at c = 3e8 m/s. The intrinsic "
            "impedance of free space is approximately 377 ohms. The power density of a plane wave "
            "falls as 1/r^2 with distance (the inverse square law).",
        ]),
    ]),
    ("07_microprocessors", "Microprocessors and Embedded Systems - Lecture Notes", [
        ("1. Microprocessor Architecture", [
            "A microprocessor executes instructions in a fetch-decode-execute cycle. The program "
            "counter holds the address of the next instruction, the instruction register holds the "
            "instruction being decoded, and the accumulator holds intermediate results. The "
            "arithmetic logic unit (ALU) performs arithmetic and logic operations, and the status "
            "(flag) register records conditions such as zero, carry and overflow.",
        ]),
        ("2. The 8051 Microcontroller", [
            "The 8051 is an 8-bit microcontroller with 4 KB of on-chip program ROM and 128 bytes "
            "of RAM. It has 32 general-purpose I/O pins arranged in four 8-bit ports (P0 to P3), "
            "two 16-bit timers (Timer 0 and Timer 1), and one serial port (UART). It uses 40 pins "
            "in a dual in-line package (DIP40).",
            "Timer modes: Mode 0 is 13-bit, Mode 1 is 16-bit, Mode 2 is 8-bit auto-reload and "
            "Mode 3 is split timer. The machine cycle of the 8051 is 12 oscillator periods; with a "
            "12 MHz crystal the machine cycle is 1 microsecond.",
        ]),
        ("3. Memory and Addressing", [
            "Von Neumann architecture uses a single memory space for code and data; Harvard "
            "architecture uses separate program and data memories. The 8051 uses a Harvard "
            "architecture. Addressing modes include immediate, direct, register, register "
            "indirect and indexed (used with the data pointer DPTR for look-up tables).",
            "Cache memory sits between the CPU and main memory to reduce average access time. "
            "Common cache mapping schemes are direct-mapped, fully associative and set "
            "associative.",
        ]),
    ]),
    ("08_digital_signal_processing", "Digital Signal Processing - Lecture Notes", [
        ("1. Discrete Signals and the Z-Transform", [
            "A discrete-time signal x[n] is defined at integer n. The Z-transform X(z) = sum over n "
            "of x[n] z^(-n). The region of convergence determines uniqueness. The Z-transform "
            "converts convolution into multiplication: if y[n] = x[n] * h[n] then Y(z) = X(z)H(z), "
            "and the transfer function is H(z).",
            "A causal system has a right-sided impulse response. For a stable causal system all "
            "poles of H(z) lie inside the unit circle |z| = 1.",
        ]),
        ("2. The Discrete Fourier Transform", [
            "The N-point Discrete Fourier Transform is X[k] = sum from n=0 to N-1 of "
            "x[n] e^(-j*2*pi*k*n/N), for k = 0 to N-1. The DFT is periodic with period N and "
            "relates to the DFT of a signal sampled at fs, where bin k corresponds to frequency "
            "k*fs/N.",
            "The frequency resolution of an N-point DFT is fs/N. Zero-padding interpolates the "
            "spectrum without increasing the true resolution. Spectral leakage occurs because a "
            "finite record truncates the signal; windowing (Hamming, Hann, Blackman) reduces "
            "leakage at the cost of widening the main lobe.",
        ]),
        ("3. The Fast Fourier Transform", [
            "The Fast Fourier Transform (FFT) computes the DFT with complexity O(N log N) instead "
            "of the direct O(N^2). The radix-2 Cooley-Tukey algorithm requires N to be a power of "
            "two and splits the transform into even and odd indexed sub-transforms recursively.",
            "For N = 1024 the FFT needs about 10*N operations instead of N^2, roughly a "
            "hundred-fold saving.",
        ]),
        ("4. Digital Filters", [
            "Finite Impulse Response (FIR) filters have an impulse response of finite length, are "
            "always stable and can be designed with exactly linear phase. Infinite Impulse "
            "Response (IIR) filters use feedback, need fewer coefficients for the same magnitude "
            "response but have non-linear phase and can be unstable.",
            "A moving-average filter of length M is y[n] = (1/M) * sum of the last M inputs; it is "
            "a simple low-pass FIR filter. The bilinear transform maps an analogue prototype "
            "filter to the digital domain and is used to design IIR filters such as Butterworth "
            "and Chebyshev filters.",
        ]),
    ]),
]


def build_one(stem: str, title: str, sections: list[tuple[str, list[str]]]) -> Path:
    out = HERE / f"{stem}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=title)
    flow = [Paragraph(title, H1), Spacer(1, 2)]
    for head, paras in sections:
        flow.append(Paragraph(head, H2))
        for para in paras:
            flow.append(Paragraph(para, BODY))
    doc.build(flow)
    return out


def main() -> None:
    total = 0
    for stem, title, sections in DOCS:
        path = build_one(stem, title, sections)
        n = sum(len(p) for _, p in sections)
        total += n
        print(f"wrote {path.name:38s} {n:2d} paragraphs")
    print(f"\n{len(DOCS)} documents, {total} paragraphs -> {HERE}")


if __name__ == "__main__":
    main()
