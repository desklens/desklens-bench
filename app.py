{
  "type": "object",
  "properties": {
    "call_category": {
      "type": "string",
      "description": "patient_call: a patient or family member about a visit, symptom, appointment, medicine, test, report or bill. clinical_consultation: two clinicians discussing one specific patient's treatment, no patient on the line. operations_call: staff and doctors coordinating schedules, arrivals, no-shows or availability, no patient or family on the line and no treatment discussed. non_patient_call: vendor, supplier, seller, job enquiry, wrong number or personal call, with no specific patient referenced.",
      "enum": [
        "patient_call",
        "clinical_consultation",
        "operations_call",
        "non_patient_call"
      ]
    },
    "speaker_roles": {
      "type": "string",
      "description": "One short sentence naming what each side did, so a reader can check the sides are the right way round, e.g. 'caller described tooth pain and asked cost; clinic offered a slot'. If the sides cannot be told apart, write 'sides unclear'."
    },
    "call_type": {
      "type": "string",
      "description": "What the call was about. procedure_inquiry: cost, duration or number of sittings of a treatment. billing_payment: an existing bill, money owed, a payment made or a refund. general_inquiry: timings, address or staff availability only. report_status: whether a report is ready. test_results: the content of a result already available. prescription_refill: medicine finished, needs more. follow_up: a return visit after earlier treatment. Use other only when nothing fits.",
      "enum": [
        "appointment_booking",
        "reschedule",
        "cancellation",
        "procedure_inquiry",
        "prescription_refill",
        "dosage_adjustment",
        "report_status",
        "test_results",
        "billing_payment",
        "complaint",
        "general_inquiry",
        "emergency",
        "follow_up",
        "treatment_planning",
        "other"
      ]
    },
    "urgency": {
      "type": "string",
      "description": "routine unless the caller names a symptom on the EMERGENCY or URGENT list in the specialty block, or speaks words like emergency, urgent, immediately, right now. Never judged from tone or volume.",
      "enum": [
        "emergency",
        "urgent",
        "routine"
      ]
    },
    "patient_name": {
      "type": "string",
      "description": "The patient's own name wherever it appears, including fragments like 'for Harini'. If an unnamed patient such as an infant is recorded under a guardian's name, use that name. Exclude a name the transcript shows belongs to someone who is not the patient. Never a company name.",
      "nullable": true
    },
    "patient_age": {
      "type": "string",
      "description": "Spoken age converted to 'X years Y months'. Null if not stated.",
      "nullable": true
    },
    "caller_is_patient": {
      "type": "boolean",
      "description": "true when the patient speaks for themselves, false when someone calls on their behalf, null when unclear or no patient is involved.",
      "nullable": true
    },
    "patient_status": {
      "type": "string",
      "description": "returning when the caller refers to an earlier visit, existing treatment or previous doctor contact. new when they say it is their first time. Otherwise unclear.",
      "enum": [
        "new",
        "returning",
        "unclear"
      ]
    },
    "symptoms_mentioned": {
      "type": "array",
      "description": "Short terms of two to four words in the caller's everyday wording, e.g. 'black dots on face'. Never a copied sentence. Each symptom once. Empty if none.",
      "items": {
        "type": "string"
      }
    },
    "symptom_duration": {
      "type": "string",
      "description": "How long the problem has existed, exactly as stated, e.g. 'two days'. Null if not said.",
      "nullable": true
    },
    "procedures_mentioned": {
      "type": "array",
      "description": "Short terms of two to four words, each procedure once. Only what was actually spoken. Empty if none.",
      "items": {
        "type": "string"
      }
    },
    "medications_mentioned": {
      "type": "array",
      "description": "Original drug names as spoken, each once. Empty if none.",
      "items": {
        "type": "string"
      }
    },
    "tests_mentioned": {
      "type": "array",
      "description": "Short terms of two to four words, each test once. Empty if none.",
      "items": {
        "type": "string"
      }
    },
    "appointment": {
      "type": "object",
      "description": "The visit agreed on THIS call only.",
      "properties": {
        "action": {
          "type": "string",
          "description": "booked only when both sides agree a specific date or time on this call. requested when intent to come is stated with no slot fixed. none when there is no intent to visit. A previously scheduled time that has passed or failed never counts.",
          "enum": [
            "booked",
            "rescheduled",
            "cancelled",
            "requested",
            "none"
          ]
        },
        "date": {
          "type": "string",
          "description": "Only a date agreed on this call. Clinic opening hours and past or failed slots are never appointment dates. Must be null unless action is booked or rescheduled.",
          "nullable": true
        },
        "time": {
          "type": "string",
          "description": "Only a time agreed on this call. Clinic opening hours and past or failed slots are never appointment times. Must be null unless action is booked or rescheduled.",
          "nullable": true
        }
      },
      "required": [
        "action",
        "date",
        "time"
      ],
      "propertyOrdering": [
        "action",
        "date",
        "time"
      ]
    },
    "walk_in_suggested": {
      "type": "boolean",
      "description": "true when the clinic tells the caller to just come without fixing a time."
    },
    "price_asked": {
      "type": "boolean",
      "description": "true when the caller asks what a clinic treatment or service costs, in any wording. A seller quoting their own product never sets this."
    },
    "price_quoted": {
      "type": "boolean",
      "description": "true only when the clinic states an actual figure or range. 'Doctor will tell after seeing' and 'come for consultation first' are deflections, not quotes."
    },
    "price_amount": {
      "type": "string",
      "description": "The figure exactly as stated, if one was given. Null otherwise.",
      "nullable": true
    },
    "who_asked_price": {
      "type": "string",
      "description": "Which side raised cost. Null if nobody asked.",
      "nullable": true,
      "enum": [
        "caller",
        "clinic",
        null
      ]
    },
    "who_deflected_price": {
      "type": "string",
      "description": "clinic when the clinic avoided giving a figure after being asked. Null when a figure was given or nobody asked.",
      "nullable": true,
      "enum": [
        "caller",
        "clinic",
        null
      ]
    },
    "non_conversion_reason": {
      "type": "string",
      "description": "Filled ONLY for patient_call when the caller wanted a visit and the call ended WITHOUT a booking AND without the caller stating they will come. If the caller says they will come, or a walk-in was suggested and accepted, this stays null - the caller is not lost. Always null for operations_call, clinical_consultation and non_patient_call, and null when a booking happened or no visit was ever in question.",
      "nullable": true,
      "enum": [
        "no_slot_available",
        "price_not_given",
        "price_too_high",
        "doctor_unavailable",
        "patient_deferred",
        "referred_elsewhere",
        "information_only",
        "unclear",
        null
      ]
    },
    "flags": {
      "type": "array",
      "description": "The first thing a doctor looks at. Add only when the words were actually spoken. possible_emergency: a symptom on the specialty EMERGENCY list, or emergency words. patient_upset: any caller expresses anger or frustration, including a doctor or staff member. urgent_callback: someone asks to be called back urgently. payment_pending: money stated as still owed. complaint_unresolved: a complaint ends without resolution. comprehension_issue: someone says they do not understand or cannot name what they mean.",
      "items": {
        "type": "string",
        "enum": [
          "possible_emergency",
          "patient_upset",
          "urgent_callback",
          "payment_pending",
          "complaint_unresolved",
          "comprehension_issue"
        ]
      }
    },
    "action_items": {
      "type": "array",
      "description": "Only what someone actually committed to on the call, as short instructions naming the committing side, e.g. 'Clinic to call back with the rate'. Empty if nothing was promised.",
      "items": {
        "type": "string"
      }
    },
    "outcome": {
      "type": "string",
      "description": "resolved: the caller's question got a direct answer or the action was confirmed done. pending_action: an answer was given but someone must still act. needs_callback: a call back was explicitly promised. escalated: passed to someone more senior during the call. unresolved: the call ended without the answer it was made for, even if someone said they would follow up later.",
      "enum": [
        "resolved",
        "pending_action",
        "needs_callback",
        "escalated",
        "unresolved"
      ]
    },
    "one_line_summary": {
      "type": "string",
      "description": "One sentence in your own words: who called, what they wanted, what happened. Never copied from the transcript."
    },
    "detailed_summary": {
      "type": "string",
      "description": "Null for routine calls where the one line says everything. Two to four sentences only when there is clinical detail, a complaint, a medicine or dose change, a treatment plan, or something unresolved. Built from the extracted fields, in simple English, keeping names and medicine names as spoken. Never copied from the transcript.",
      "nullable": true
    },
    "confidence": {
      "type": "string",
      "description": "Filled LAST, after every other field. A verdict on the whole transcript: low if there were garbled or impossible lines anywhere, if the sides talked past each other, or if who-is-who was uncertain - even when the main facts came out clearly. high means the transcript read cleanly end to end. When in doubt, low.",
      "enum": [
        "high",
        "low"
      ]
    }
  },
  "propertyOrdering": [
    "call_category",
    "speaker_roles",
    "call_type",
    "urgency",
    "patient_name",
    "patient_age",
    "caller_is_patient",
    "patient_status",
    "symptoms_mentioned",
    "symptom_duration",
    "procedures_mentioned",
    "medications_mentioned",
    "tests_mentioned",
    "appointment",
    "walk_in_suggested",
    "price_asked",
    "price_quoted",
    "price_amount",
    "who_asked_price",
    "who_deflected_price",
    "non_conversion_reason",
    "flags",
    "action_items",
    "outcome",
    "one_line_summary",
    "detailed_summary",
    "confidence"
  ],
  "required": [
    "call_category",
    "speaker_roles",
    "call_type",
    "urgency",
    "patient_name",
    "patient_age",
    "caller_is_patient",
    "patient_status",
    "symptoms_mentioned",
    "symptom_duration",
    "procedures_mentioned",
    "medications_mentioned",
    "tests_mentioned",
    "appointment",
    "walk_in_suggested",
    "price_asked",
    "price_quoted",
    "price_amount",
    "who_asked_price",
    "who_deflected_price",
    "non_conversion_reason",
    "flags",
    "action_items",
    "outcome",
    "one_line_summary",
    "detailed_summary",
    "confidence"
  ]
}
