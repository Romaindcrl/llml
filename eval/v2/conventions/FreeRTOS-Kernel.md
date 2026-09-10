# FreeRTOS — Coding Standard and Style Guide

> Verbatim : « Mastering the FreeRTOS Real Time Kernel », ch. 2 §2.5,
> repo officiel FreeRTOS/FreeRTOS-Kernel-Book @ba20b2f82092 (snapshot 2026-07-11).
> Complément mécanique : .github/uncrustify.cfg du kernel @9db704cd.

## 2.5 Data Types and Coding Style Guide

### 2.5.1 Data Types

Each port of FreeRTOS has a unique portmacro.h header file that contains
(amongst other things) definitions for two port-specific data types:
`TickType_t` and `BaseType_t`. The following list describes the macro or
typedef used and the actual type:

- `TickType_t`

  FreeRTOS configures a periodic interrupt called the tick interrupt.

  The number of tick interrupts that have occurred since the FreeRTOS
  application started is called the *tick count*. The tick count is
  used as a measure of time.

  The time between two tick interrupts is called the *tick period*.
  Times are specified as multiples of tick periods.

  `TickType_t` is the data type used to hold the tick count value, and to
  specify times.

  `TickType_t` can be an unsigned 16-bit type, an unsigned 32-bit
  type, or an unsigned 64-bit type, depending on the setting of `configTICK_TYPE_WIDTH_IN_BITS`
  in FreeRTOSConfig.h. The setting of `configTICK_TYPE_WIDTH_IN_BITS` is architecture
  dependent. FreeRTOS ports will also check if this setting is valid.

  Using a 16-bit type can greatly improve efficiency on 8-bit and
  16-bit architectures, but severely limits the maximum block time that
  can be specified in FreeRTOS API calls. There is no reason to use a
  16-bit `TickType_t` type on a 32-bit or 64-bit architecture.

  Previous use of `configUSE_16_BIT_TICKS` has been replaced by `configTICK_TYPE_WIDTH_IN_BITS` to support
  tick counts greater than 32-bits. New designs should use `configTICK_TYPE_WIDTH_IN_BITS`
  instead of `configUSE_16_BIT_TICKS`.

   <a name="tbl2" title="Table 2 TickType_t data type and the configTICK_TYPE_WIDTH_IN_BITS configuration"></a>

   * * *
   | configTICK\_TYPE\_WIDTH\_IN\_BITS | 8-bit architectures | 16-bit architectures | 32-bit architectures | 64-bit architectures |
   | --- | --- | --- | --- | --- |
   | TICK\_TYPE\_WIDTH\_16_BITS | uint16\_t	| uint16\_t	| uint16\_t	| N/A |
   | TICK\_TYPE\_WIDTH\_32_BITS | uint32\_t	| uint32\_t	| uint32\_t	| N/A |
   | TICK\_TYPE\_WIDTH\_64_BITS | N/A | N/A | uint64\_t | uint64\_t |

   ***Table 2*** *TickType_t data type and the configTICK_TYPE_WIDTH_IN_BITS configuration*
   * * *

- `BaseType_t`

  This is always defined as the most efficient data type for the
  architecture. Typically, this is a 64-bit type on a 64-bit architecture,
  a 32-bit type on a 32-bit architecture, a 16-bit type on a 16-bit
  architecture, and an 8-bit type on an 8-bit architecture.

  `BaseType_t` is generally used for return types that take only a very
  limited range of values, and for `pdTRUE`/`pdFALSE` type Booleans.


*List of port specific data types used by FreeRTOS*


### 2.5.2 Variable Names

Variables are prefixed with their type: 'c' for `char`, 's' for `int16_t`
(short), 'l' for `int32_t` (long), and 'x' for `BaseType_t` and any other
non-standard types (structures, task handles, queue handles, etc.).

If a variable is unsigned, it is also prefixed with a 'u'. If a variable
is a pointer, it is also prefixed with a 'p'. For example, a variable of
type `uint8_t` will be prefixed with 'uc', and a variable of type pointer
to char (`char *`) will be prefixed with 'pc'.


### 2.5.3 Function Names

Functions are prefixed with both the type they return and the file they
are defined within. For example:

- v**Task**PrioritySet() returns a *v*oid and is defined within **tasks**.c.
- x**Queue**Receive() returns a variable of type *BaseType_t* and is defined within **queue**.c.
- pv**Timer**GetTimerID() returns a *p*ointer to *v*oid and is defined within **timers**.c.

File scope (private) functions are prefixed with 'prv'.


### 2.5.4 Formatting

Tabs are used in some demo applications where one tab is always set to
equal four spaces. The kernel no longer uses tabs.


### 2.5.5 Macro Names

Most macros are written in upper case, and prefixed with lower case
letters that indicate where the macro is defined. Table 3 provides a
list of prefixes.

<a name="tbl3" title="Table 3 Macro prefixes"></a>

* * *
| Prefix                                       | Location of macro definition   |
|----------------------------------------------|--------------------------------|
| port (for example, `portMAX_DELAY`)          | `portable.h` or `portmacro.h`  |
| task (for example, `taskENTER_CRITICAL()`)   | `task.h`                       |
| pd (for example, `pdTRUE`)                   | `projdefs.h`                   |
| config (for example, `configUSE_PREEMPTION`) | `FreeRTOSConfig.h`             |
| err (for example, `errQUEUE_FULL`)           | `projdefs.h`                   |

***Table 3*** *Macro prefixes*
* * *

Note that the semaphore API is written almost entirely as a set of
macros, but follows the function naming convention, rather than the
macro naming convention.

The macros defined in Table 4 are used throughout the FreeRTOS source code.

<a name="tbl4" title="Table 4 Common macro definitions"></a>

* * *
| Macro        | Value |
|--------------|-------|
| `pdTRUE`     | 1     |
| `pdFALSE`    | 0     |
| `pdPASS`     | 1     |
| `pdFAIL`     | 0     |

***Table 4*** *Common macro definitions*
* * *


### 2.5.6 Rationale for Excessive Type Casting

The FreeRTOS source code compiles with many different compilers, many of
which differ in how and when they generate warnings. In particular,
different compilers want casting used in different ways. As a result,
the FreeRTOS source code contains more type casting than would normally
be warranted.
